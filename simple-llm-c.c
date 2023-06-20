/*
  Description: C program for generating phrases using Markov Chain
  An extremely simple example of Markov Chain application.
  Author: Allan Barcelos <allan@barcelos.dev>
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LEN 100
#define ORDER 2
#define MAX_WORDS 1000

typedef struct Node {
    char word[MAX_LEN];
    struct Node* next;
} Node;

Node* start;

void insertWord(char* word) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    strcpy(newNode->word, word);
    newNode->next = NULL;

    if (start == NULL) {
        start = newNode;
    } else {
        Node* temp = start;
        while (temp->next != NULL) {
            temp = temp->next;
        }
        temp->next = newNode;
    }
}

void generatePhrase(int numWords) {
    Node* current = start;

    if (current == NULL) {
        printf("There is not enough data to generate phrases.\n");
        return;
    }

    // Next random initial word
    int count = rand() % MAX_WORDS;
    for (int i = 0; i < count; i++) {
        if (current->next == NULL) {
            current = start;
        } else {
            current = current->next;
        }
    }

    // generate phrase
    for (int i = 0; i < numWords; i++) {
        printf("%s ", current->word);
        if (current->next == NULL) {
            current = start;
        } else {
            current = current->next;
        }
    }
    printf("\n");
}

int main() {
    srand(time(0));

    char input[MAX_LEN];
    char dataset[MAX_WORDS][MAX_LEN];
    int numWords = 0;

    printf("Enter the training phrases (type 'exit' to finish):\n");

    while (1) {
        printf("> ");
        fgets(input, sizeof(input), stdin);
        input[strlen(input) - 1] = '\0';  // remove new line chracter

        if (strcmp(input, "exit") == 0) {
            break;
        }

        strcpy(dataset[numWords], input);
        numWords++;
    }

    // build Markov Chain
    for (int i = 0; i < numWords; i++) {
        char* token = strtok(dataset[i], " ");
        char* prevWords[ORDER];

        for (int j = 0; j < ORDER; j++) {
            prevWords[j] = (char*)malloc(MAX_LEN * sizeof(char));
            strcpy(prevWords[j], "");
        }

        while (token != NULL) {
            // insert current word in chain
            char currentWord[MAX_LEN];
            strcpy(currentWord, token);
            insertWord(currentWord);

            // update the older words
            for (int j = 0; j < ORDER - 1; j++) {
                strcpy(prevWords[j], prevWords[j + 1]);
            }
            strcpy(prevWords[ORDER - 1], currentWord);

            token = strtok(NULL, " ");
        }
    }

    printf("\nEnter the number of words to generate a phrase: ");
    scanf("%d", &numWords);

    generatePhrase(numWords);

    return 0;
}

