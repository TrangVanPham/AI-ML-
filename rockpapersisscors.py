import random

def play_game():
    # Computer's random choice
    computer_choices = ["rock", "paper", "scissors"]
    computer_play = random.choice(computer_choices)

    # Player's input
    player_play = input("Enter your choice (rock, paper, scissors): ").lower()

    # Check for invalid input
    if player_play not in computer_choices:
        print("Error: Invalid input! Please choose 'rock', 'paper', or 'scissors'.")
        return

    # Determine the winner
    print(f"\nComputer chose: {computer_play}")
    print(f"You chose: {player_play}\n")

    if player_play == computer_play:
        print("It's a tie!")
    elif (
        (player_play == "rock" and computer_play == "scissors") or
        (player_play == "paper" and computer_play == "rock") or
        (player_play == "scissors" and computer_play == "paper")
    ):
        print("You win!")
    else:
        print("Computer wins!")

# Run the game
if __name__ == "__main__":
    play_game()