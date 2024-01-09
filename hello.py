import random

def player(prev_play, opponent_history = []):
    if prev_play == "":
        # Choose randomly for the first move
        return random.choice(["R", "P", "S"])
    
    # Calculate the next move based on the previous move of the opponent
    # Here, we can use multiple strategies and even change them based on the history of the opponent's moves
    
    # For example, let's create a simple strategy that chooses the move that beats the previous move of the opponent
    if prev_play == "R":
        return "P"
    elif prev_play == "P":
        return "S"
    else:
        return "R"