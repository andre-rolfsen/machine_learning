# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <string.h>

#define MAX_LINHA 1024
# define N 1001
# define DEGREE 2
# define EPOCHS 5000
# define ALPHA 0.01f

float relu(float x) {
    if (x < 0) return 0;
    return x;
}

float dRelu(float x) {
    return x > 0 ? 1 : 0;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float predict(float x, int n, float w[DEGREE][2]) {
    float sum = 0;
    for (int j = 0; j < DEGREE; j++) {
        sum += pow(x,j)*w[j][n];
    }
    return sum;
}

float J(float y, float prev) {
    float eps = 1e-7f;

    if (prev < eps) prev = eps;
    if (prev > 1.0f - eps) prev = 1.0f - eps;

    return -(y * logf(prev) + (1.0f - y) * logf(1.0f - prev));
}


float forward(float x, float w[DEGREE][2], float z[2], float a[2]) {
    z[0] = predict(x, 0, w);
    a[0] = relu(z[0]);

    z[1] = predict(a[0], 1, w);
    a[1] = sigmoid(z[1]);

    return a[1];
}

void backward(
    float x,
    float y,
    float w[2][2],
    float z[2],
    float a[2],
    float delta[2],
    float grad[2][2],
    float lr
) {

    // erro para y
    delta[1] = a[1] - y;

    // calculo de gradiente(coeficiente angular) de saida
    grad[1][1] = delta[1] * a[0];
    // calculo de gradiente(coeficiente angular) de entrada
    grad[0][1] = delta[1];
    
    delta[0] = delta[1] * w[1][1] * dRelu(z[0]);

    // calculo de gradiente(coeficiente linear) de saida
    grad[1][0] = delta[0] * x;
    // calculo de gradiente(coeficiente linear) de entrada
    grad[0][0] = delta[0];


    for (int i = 0; i < 2; i++) {
        w[i][0] -= lr * grad[i][0];
        w[i][1] -= lr * grad[i][1];
    }
}

float table[N][2];
int main() {
    FILE *arquivo = fopen("StudentsPerformance.csv", "r");

    if (arquivo == NULL) {
        printf("Erro: nao foi possivel abrir o arquivo students.csv\n");
        return 1;
    }

    char linha[MAX_LINHA];

    if (fgets(linha, sizeof(linha), arquivo) == NULL) {
        printf("Erro: arquivo vazio.\n");
        fclose(arquivo);
        return 1;
    }

    int i = 0;

    while (fgets(linha, sizeof(linha), arquivo) != NULL && i < N) {

        linha[strcspn(linha, "\r\n")] = '\0';

        linha[strcspn(linha, "\r\n")] = '\0';

        char *gender     = strtok(linha, "\",");
        char *race       = strtok(NULL, "\",");
        char *educacao   = strtok(NULL, "\",");
        char *almoco     = strtok(NULL, "\",");
        char *preparacao = strtok(NULL, "\",");
        char *math       = strtok(NULL, "\",");
        char *reading    = strtok(NULL, "\",");
        char *writing    = strtok(NULL, "\",");

        if (gender == NULL || race == NULL || educacao == NULL ||
            almoco == NULL || preparacao == NULL || math == NULL ||
            reading == NULL || writing == NULL) {
            printf("Linha invalida: %s\n", linha);
            continue;
            }

        table[i][0] = (float)atoi(math)/100;
        table[i][1] = strcmp("male", gender) == 0 ? 1:0;
        char classific[7];
        strcpy(classific, (table[i][1] == 1) ? "male" : "female");

        printf("Aluno %d: math = %f, gender = %s\n",
               i + 1, table[i][0], classific);
        i++;
    }

    fclose(arquivo);

    printf("\nTotal de linhas carregadas: %d\n", i);

    printf("\n");

    float w[DEGREE][2];
    float z[2];
    float a[2];
    float delta[2] = {0};
    float grad[2][2] = {0};
    float x;
    float y;
    float prev;
    float Jmin = INFINITY;
    float argmin[DEGREE][2];
    float loss = 0;

    for (int i = 0; i < DEGREE; i++) {
        w[i][0] = ((float) rand() / RAND_MAX - 0.5f) * 0.2f;
        w[i][1] = ((float) rand() / RAND_MAX - 0.5f) * 0.2f;
    }


    for (int i=0; i<EPOCHS; i++) {
        loss =0;
        for (int j = 0; j < N; j++) {
            x = table[j][0];
            y = table[j][1];
            prev = forward(x, w, z, a);

            loss += J(y,prev);

            backward(x, y, w, z, a, delta, grad, ALPHA);
        }

        loss /= N;

        if (loss < Jmin) {
            Jmin = loss;
            for (int j = 0; j < DEGREE; j++) {
                argmin[0][j] = w[0][j];
                argmin[1][j] = w[1][j];
            }
        };

    }

    printf("BCE=%F\n", Jmin);
    printf("X=");
    scanf("%f",&x);
    printf("Y= %d", forward(x, argmin, z, a) > 0.5 ? 1 : 0);
}