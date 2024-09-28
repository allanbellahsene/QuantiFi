class SharedState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedState, cls).__new__(cls)
            cls._instance.OPEN_ORDERS = {}
        return cls._instance

    def get_open_orders(self):
        return self.OPEN_ORDERS

    def set_open_orders(self, orders):
        self.OPEN_ORDERS = orders

    def update_open_orders(self, new_orders):
        self.OPEN_ORDERS.update(new_orders)

    def clear_open_orders(self):
        self.OPEN_ORDERS.clear()

shared_state = SharedState()

def get_shared_state():
    return shared_state