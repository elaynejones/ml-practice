suits = ["Hearts", "Spades", "Diamonds", "Clubs"]
ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
deck =[]

for suit in suits:
    for rank in ranks:
        deck.append(f'{rank} of {suit}') 

deck_count= len(deck)

print(f"There are {deck_count} cards in the deck.")

print("Dealing...")

hand = []

import random


while len(hand)<5:
    deal = random.choice(deck)
    hand.append(deal)
    deck.remove(deal) 

    

new_deck = len(deck)

print (f"There are {new_deck} cards in the deck.")
print(f"Player has the following cards in their hand: \n {hand}")
