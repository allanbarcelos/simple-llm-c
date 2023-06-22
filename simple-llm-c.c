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

typedef struct MarkovChain {
    int count;
    struct Node* nextWords;
} MarkovChain;

MarkovChain* markovChain;

void insertWord(char* word) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    strcpy(newNode->word, word);
    newNode->next = NULL;

    if (markovChain->nextWords == NULL) {
        markovChain->nextWords = newNode;
    } else {
        Node* temp = markovChain->nextWords;
        while (temp->next != NULL) {
            temp = temp->next;
        }
        temp->next = newNode;
    }

    markovChain->count++;
}

char* chooseNextWord() {
    int choice = rand() % markovChain->count;
    Node* current = markovChain->nextWords;

    while (choice > 0) {
        current = current->next;
        choice--;
    }

    return current->word;
}

void generateCompletion() {
    if (markovChain->count == 0) {
        printf("There is not enough data to generate phrases.\n");
        return;
    }

    printf("Type the beginning of a phrase: ");
    char input[MAX_LEN];
    fgets(input, sizeof(input), stdin);
    input[strlen(input) - 1] = '\0'; // remove new line character

    Node* current = markovChain->nextWords;
    char* token = strtok(input, " ");

    while (token != NULL) {
        printf("%s ", token);

        // Update current word for generating completion
        strcpy(current->word, token);

        if (current->next == NULL) {
            current = markovChain->nextWords;
        } else {
            current = current->next;
        }

        token = strtok(NULL, " ");
    }

    printf("\nGenerated completion: %s ", input);

    // Generate the completion
    for (int i = 0; i < ORDER; i++) {
        printf("%s ", current->word);

        if (current->next == NULL) {
            current = markovChain->nextWords;
        } else {
            current = current->next;
        }
    }

    printf("\n");
}


void trainingFromInput() {
    char input[MAX_LEN];
    char dataset[MAX_WORDS][MAX_LEN];
    int numWords = 0;

    printf("Enter the training phrases (type 'exit' to finish):\n");

    while (1) {
        printf("> ");
        fgets(input, sizeof(input), stdin);
        input[strlen(input) - 1] = '\0';  // remove new line character

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

    printf("Training completed successfully.\n");
}

void trainingFromFile() {
    char filename[MAX_LEN];
    printf("Enter the filename with training phrases: ");
    scanf("%s", filename);

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return;
    }

    char line[MAX_LEN];
    while (fgets(line, sizeof(line), file)) {
        line[strlen(line) - 1] = '\0'; // remove new line character
        char* token = strtok(line, " ");
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

    fclose(file);
    printf("Training completed successfully.\n");
}

int main() {
    srand(time(0));

    int choice;
    markovChain = (MarkovChain*)malloc(sizeof(MarkovChain));
    markovChain->count = 0;
    markovChain->nextWords = NULL;

    do {
        printf("Menu:\n");
        printf("1. Training from input\n");
        printf("2. Training from file\n");
        printf("3. Type and Generate the Completion\n");
        printf("4. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                trainingFromInput();
                break;
            case 2:
                trainingFromFile();
                break;
            case 3:
                generateCompletion();
                break;
            case 4:
                printf("Exiting program.\n");
                break;
            default:
                printf("Invalid choice. Please try again.\n");
                break;
        }

        printf("\n");
    } while (choice != 4);

    // Free memory
    Node* current = markovChain->nextWords;
    while (current != NULL) {
        Node* temp = current;
        current = current->next;
        free(temp);
    }

    free(markovChain);

    return 0;
}
