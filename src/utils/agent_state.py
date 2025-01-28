class AgentState:
    def __init__(self):
        self._stop_requested = False
    
    def request_stop(self):
        self._stop_requested = True
    
    def clear_stop(self):
        self._stop_requested = False
    
    def is_stop_requested(self):
        return self._stop_requested