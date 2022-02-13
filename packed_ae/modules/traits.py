

class HasTaskID():
    """Trait that sets up task ids"""
    _task_id: int = 0

    def on_task_change(self, new_task_id: int):
        pass
    
    @property
    def task(self) -> int:
        return self._task_id

    @task.setter
    def task(self, value: int):
        self.on_task_change(value)
        self._task_id = value

    @property
    def task_tag(self) -> str:
        return f"{self.task:04d}"
