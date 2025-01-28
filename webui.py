import os
import json
import asyncio
import re
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from langchain_ollama import ChatOllama

###################################################
# 1. Load environment variables from .env
###################################################
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-r1:7b")
CHROME_PATH = os.getenv("CHROME_PATH", None)

###################################################
# 2. Initialize Ollama with environment settings
###################################################
try:
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL_NAME,
        temperature=1
    )
    print(f"✅ Connected to Ollama at {OLLAMA_BASE_URL} with model '{LLM_MODEL_NAME}'")
except Exception as e:
    print(f"❌ Failed to connect to Ollama: {e}")
    exit(1)

###################################################
# 3. Define the schema for browser actions
###################################################
class BrowserAction(BaseModel):
    action_type: str = Field(..., description="Type of action (navigate, click, type, press, wait)")
    selector: Optional[str] = Field(None, description="CSS selector for the element")
    value: Optional[str] = Field(None, description="Value to type or URL to navigate to")
    description: str = Field(..., description="Human-readable description of the action")

###################################################
# 4. Generate actions using Ollama
###################################################
async def generate_actions(task: str, max_retries: int = 3) -> List[BrowserAction]:
    """
    Given a user task, ask Ollama to produce a list of JSON instructions,
    then parse them into Python objects (BrowserAction).
    """
    # Pydantic v2 approach:
    schema_str = json.dumps(BrowserAction.model_json_schema(), indent=2)

    # Refined prompt to ensure only JSON is returned
    prompt = f"""You are a professional web automation assistant. 
Given this task:

{task}

1. Break it down into sequential browser actions.
2. Use only the following valid action types:
   - navigate(url): Go to a URL
   - click(selector): Click an element
   - type(selector, text): Type text into an element
   - press(key): Press a keyboard key
   - wait(seconds): Wait for N seconds

you can **only** respond with a JSON array of actions following this format, DON'T REPLY ANYTHING ELSE THAN THE ARRAY:
[
  {{ 
    "action_type": "navigate",
    "value": "https://example.com",
    "description": "Navigate to example site"
  }},
  {{ 
    "action_type": "type",
    "selector": "input.search",
    "value": "query",
    "description": "Type search query"
  }}
]

Important:
- Use square brackets [ ] for the array
- Never use curly braces {{ }} as the root element
- Never include markdown code blocks
- Never include <think> tags or explanations
- Don't reply anything else than the array

{schema_str}
"""

    print("\n🔍 Sending prompt to Ollama:")
    print(prompt)

    try:
        response = await llm.ainvoke(prompt)
        print("\n📥 Received response from Ollama:")
        print(response)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to get response from Ollama: {e}")

    # Check if response.content exists and is not empty
    if not hasattr(response, 'content') or not response.content.strip():
        raise ValueError("AI response is empty.")

    # Attempt to parse the response as JSON
    try:
        response_content = response.content.strip()

        # Remove <think> tags if present
        response_content = re.sub(r'</?think>', '', response_content).strip()

        # Extract JSON array within ```json ... ``` code blocks
        json_match = re.search(r'```json\s*(\[.*?\]|{.*?})\s*```', response_content)
        if json_match:
            json_content = json_match.group(1)
        else:
            # Fallback: Extract the first JSON array in the response
            json_match = re.search(r'(\[.*?\]|{.*?})', response_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
                if json_content.startswith('{'):
                    json_content = f'[{json_content}]'
            else:
                raise ValueError("No JSON array found in the AI response.")
        
        try:
            json.loads(json_content)
        except json.JSONDecodeError:
            json_content = json_content.replace("}{", "},{")
            if not json_content.startswith('['):
                json_content = f'[{json_content}]'
            
        print("\n🛠️ Attempting to parse JSON content:")
        print(json_content)

        # Now attempt to parse the cleaned JSON
        actions_data = json.loads(json_content)
        print("\n📦 Parsed JSON actions:")
        print(actions_data)

        # Map AI's action fields to BrowserAction model
        actions = []
        for action_dict in actions_data:
            browser_action = map_ai_action_to_browser_action(action_dict)
            actions.append(browser_action)

        return actions
    except json.JSONDecodeError as jde:
        print(f"\n❌ Raw JSON Content:\n{json_content}")
        raise ValueError(f"AI response is not valid JSON: {jde}\nResponse Content:\n{response.content}")
    except Exception as e:
        raise ValueError(f"AI response parsing failed: {str(e)}")

def map_ai_action_to_browser_action(action_dict: dict) -> BrowserAction:
    """
    Maps the AI's action dictionary to the BrowserAction model.
    Generates a description based on the action type and relevant fields.
    """
    action_type = action_dict.get("action_type")
    selector = action_dict.get("selector")

    value = action_dict.get("value")
    if not value:
        # Special cases for different action types
        if action_type == "navigate":
            value = action_dict.get("url")
        elif action_type == "type":
            value = action_dict.get("text")
        elif action_type == "press":
            value = action_dict.get("key")
        elif action_type == "wait":
            value = action_dict.get("seconds")

    # Generate description based on action_type
    description = action_dict.get("description", "")
    if not description:
        if action_type == "navigate":
            description = f"Navigate to {value}"
        elif action_type == "click":
            description = f"Click element: {selector}"
        elif action_type == "type":
            description = f"Type into {selector}: {value}"
        elif action_type == "press":
            description = f"Press key: {value}"
        elif action_type == "wait":
            description = f"Wait for {value} seconds"

    if action_type == "press":
        selector = None

    return BrowserAction(
        action_type=action_type,
        selector=selector,
        value=str(value) if value is not None else None,
        description=description
    )

###################################################
# 5. Execute those actions in a browser
###################################################
async def execute_actions(actions: List[BrowserAction]) -> str:
    log = []
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                headless=False,
                executable_path=CHROME_PATH if CHROME_PATH else None
            )
            context = await browser.new_context()
            await context.tracing.start(screenshots=True, snapshots=True)
            page = await context.new_page()
            await browser.close()
            print("🖥️ Browser launched successfully.")
        except Exception as e:
            await browser.close()
            raise RuntimeError(f"❌ Failed to launch browser: {e}")

        try:
            for action in actions:
                log.append(f"🚀 Executing: {action.description}")

                if action.action_type == "navigate":
                    if not action.value:
                        raise ValueError("Missing URL for navigation")
                    await page.goto(action.value, wait_until="domcontentloaded")
                    log.append(f"🌐 Navigated to: {action.value}")
                    await page.wait_for_timeout(1000)  # Additional stability wait

                elif action.action_type == "click":
                    if not action.selector:
                        raise ValueError("Missing selector for click action")
                    await page.wait_for_selector(action.selector, state="visible", timeout=10000)
                    await page.click(action.selector)
                    log.append(f"🖱️ Clicked: {action.selector}")

                elif action.action_type == "type":
                    if not action.selector or not action.value:
                        raise ValueError("Missing selector/value for typing")
                    await page.wait_for_selector(action.selector, state="visible", timeout=10000)
                    await page.fill(action.selector, action.value)
                    log.append(f"⌨️ Typed '{action.value}' into: {action.selector}")

                elif action.action_type == "press":
                    if not action.value:
                        raise ValueError("Missing key for press action")
                    await page.keyboard.press(action.value)
                    log.append(f"🔘 Pressed key: {action.value}")

                elif action.action_type == "wait":
                    if not action.value:
                        raise ValueError("Missing duration for wait")
                    await page.wait_for_timeout(float(action.value) * 1000)
                    log.append(f"⏳ Waited: {action.value}s")

                await page.wait_for_timeout(500)  # Short pause between actions

            # Save trace for debugging
            trace_path = "trace.zip"
            await context.tracing.stop(path=trace_path)
            log.append(f"🔍 Trace saved to: {trace_path}")
            
            await browser.close()
            return "\n".join(log)

        except Exception as e:
            trace_path = "error-trace.zip"
            await context.tracing.stop(path=trace_path)
            await browser.close()
            raise RuntimeError(f"""❌ Execution failed: {str(e)}
            Debugging trace saved to: {trace_path}""")

###################################################
# 6. Main function: loop until user quits
###################################################
async def main():
    print("Welcome to the AI Browser Automation Loop!")
    print("Type a task and press Enter to execute in the browser.")
    print("Type 'exit' or leave blank to quit.\n")

    while True:
        task = input("Enter your next task > ").strip()
        if not task or task.lower() == "exit":
            print("Exiting... Goodbye!")
            break

        try:
            actions = await generate_actions(task)
            print("\n📝 AI generated the following actions:")
            for act in actions:
                print(f"  - {act.action_type}({act.selector}, {act.value}) -> {act.description}")

            print("\n🛠️ Executing in the browser now...\n")
            result_log = await execute_actions(actions)
            print("======= Execution Log =======")
            print(result_log)
            print("=============================\n")

        except Exception as e:
            print(f"Error: {e}")
            print("Please try another task.\n")

###################################################
# 7. Entry point
###################################################
if __name__ == "__main__":
    asyncio.run(main())
