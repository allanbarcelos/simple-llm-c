# Simple LLM - Random Phrase Generator using Markov Chains

This C script implements a random phrase generator using Markov Chains. It reads a set of training phrases provided by the user and constructs a data structure that represents the transition probabilities between the words present in the phrases. 

## Prerequisites

- C compiler (GCC or any other compatible compiler)

## Usage Instructions

1. Compile the source code using the C compiler of your choice.
2. Run the generated program.
3. Enter the training phrases when prompted by the program. Enter "exit" to stop entering phrases.
4. Provide the number of words to generate a random phrase.
5. The program will generate and display the random phrase.

## Script Description

The `Node` structure that represents a node in the linked list. Each node contains a word and a pointer to the next node.

The global variable `start` is declared to store the start of the linked list.

The `insertWord` function is responsible for inserting a new word into the linked list.

The `generatePhrase` function generates a random phrase based on the words present in the linked list.

In the `main` function, the program starts by initializing the seed of the random number generator based on the current time.

Next, variables for user input, the training set, and the desired number of words for the generated phrase are declared.

The program prompts the user to enter the training phrases. The phrases are read one by one and stored in the training set until the user enters "exit".

After the training phrases input, the program builds the Markov Chain. For each phrase in the training set, the program breaks the phrase into words using the whitespace delimiter. It then inserts each word into the linked list and updates the previous words used to construct the chain.

Finally, the program asks the user for the number of words to generate a random phrase and calls the `generatePhrase` function with the provided number..

## Example

gcc -o simple-llm-c.o ./simple-llm-c.c && ./simple-llm-c.o