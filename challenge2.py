import random

value = random.randint(1, 10)
count = 0
guess = 0

print("Guess a number between 1 and 10")

while guess != value:
    count += 1
    guess = input(f'Enter guess #{count}: ')
    if guess.isnumeric()==False:
        print("Numbers only, please!")
        continue
    if guess.isnumeric():
        guess = int(guess)
  
    if guess<value:
        print("Your guess is too low, try again!")

    if guess>value:
        print("Your guess is too high, try again!")
else:
    print(f'You guessed it in {count} tries!')