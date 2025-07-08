import random
import math

class NumberGuesser:
    def __init__(self, min_num=1, max_num=100):
        self.min_num = min_num
        self.max_num = max_num
        self.number = random.randint(min_num, max_num)
        self.max_guesses = 4
        self.remaining_guesses = self.max_guesses
        self.max_hints = 3
        self.remaining_hints = self.max_hints
        
        # Pre-calculate hint properties
        self._calculate_factors()
        self._calculate_multiples()
        self._calculate_parity()
    
    def _calculate_factors(self):
        """Calculate all factors of the number (excluding 1 and itself)"""
        self.factors = []
        for i in range(2, int(math.sqrt(self.number)) + 1):
            if self.number % i == 0:
                if i != self.number // i:
                    self.factors.extend([i, self.number // i])
                else:
                    self.factors.append(i)
        # Remove duplicates and sort
        self.factors = sorted(list(set(self.factors)))
    
    def _calculate_multiples(self):
        """Calculate multiples of the number within the range"""
        self.multiples = []
        for i in range(2, (self.max_num // self.number) + 1):
            multiple = self.number * i
            if multiple <= self.max_num:
                self.multiples.append(multiple)
    
    def _calculate_parity(self):
        """Determine if number is even or odd"""
        self.parity = "even" if self.number % 2 == 0 else "odd"
    
    def get_hint(self):
        """Return a random hint from available hint categories"""
        if self.remaining_hints <= 0:
            return "Sorry, you've used all your hints!"
        
        self.remaining_hints -= 1
        hint_category = random.choice(['a', 'b', 'c'])
        
        if hint_category == 'a':  # Factors or multiples
            options = []
            if self.factors:
                options.append(f"One factor of my number is {random.choice(self.factors)}")
            if self.multiples:
                options.append(f"One multiple of my number is {random.choice(self.multiples)}")
            
            if not options:
                return "My number has no factors or multiples in the given range (other than itself and 1)"
            return random.choice(options)
        
        elif hint_category == 'b':  # Larger or smaller
            if self.number == self.min_num:
                larger = random.randint(self.number + 1, self.max_num)
                return f"My number is smaller than {larger}"
            elif self.number == self.max_num:
                smaller = random.randint(self.min_num, self.number - 1)
                return f"My number is larger than {smaller}"
            else:
                if random.choice([True, False]):
                    larger = random.randint(self.number + 1, self.max_num)
                    return f"My number is smaller than {larger}"
                else:
                    smaller = random.randint(self.min_num, self.number - 1)
                    return f"My number is larger than {smaller}"
        
        elif hint_category == 'c':  # Parity
            return f"My number is {self.parity}"
    
    def check_guess(self, guess):
        """Check if the guess is correct and return appropriate response"""
        self.remaining_guesses -= 1
        if guess == self.number:
            return True, f"Congratulations! You guessed the number {self.number} correctly!"
        elif guess < self.number:
            return False, "Your guess is too low."
        else:
            return False, "Your guess is too high."
    
    def get_status(self):
        """Return current game status"""
        return f"Guesses remaining: {self.remaining_guesses}, Hints remaining: {self.remaining_hints}"


def play_game():
    print("Welcome to the Number Guesser Game!")
    print(f"I'm thinking of a number between 1 and 100. You have 4 guesses and 3 hints available.")
    print("Type 'hint' if you want a hint.\n")
    
    game = NumberGuesser(1, 100)
    
    while game.remaining_guesses > 0:
        user_input = input("Enter your guess (or 'hint' for a hint): ").strip().lower()
        
        if user_input == 'hint':
            print(game.get_hint())
            print(game.get_status())
            continue
        
        try:
            guess = int(user_input)
        except ValueError:
            print("Please enter a valid number or 'hint'.")
            continue
        
        correct, message = game.check_guess(guess)
        print(message)
        
        if correct:
            return
        
        print(game.get_status())
        
        if game.remaining_guesses == 0:
            print(f"\nGame over! The number was {game.number}.")
            return
        
        print("Try again!")


if __name__ == "__main__":
    play_game()