class Solution:
    def get_slope(self, x: float) -> float:
        return 2 * x

    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        if iterations == 0:
            return init
            
        current_x = float(init)
        
        for _ in range(iterations):
            slope = self.get_slope(current_x)
            current_x = current_x - (learning_rate * slope)
            
        return round(current_x, 5)

    