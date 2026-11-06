# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <string.h>
#define MAX_LINHA 1024
# define N 1001
# define DEGREE 3
# define EPOCHS 20000
# define ALPHA 0.01f

float J(float table[N][2], float w[DEGREE + 1]) {
    float sum = 0;
    float prev = 0;
    for (int i=0; i<N; i++) {
        float x = table[i][0];
        float y = table[i][1];
        prev = 0;
        for (int j=0; j<(DEGREE +1); j++) {
            prev += w[j] * pow(x, j);
        }
        sum += (prev - y)*(prev - y);
    }
    return sum/(2*N);
}


float gradient(float table[N][2], float w[DEGREE + 1], int t) {
    float sum = 0;
    float prev = 0;
    for (int i=0; i<N; i++) {
        float x = table[i][0];
        float y = table[i][1];
        prev = 0;
        for (int j=0; j<(DEGREE + 1); j++) {
            prev += w[j] * pow(x, j);
        }
        sum += (prev - y)*pow(x,t);
    }
    return sum/(N);
}

float dpy(float table[N][2]) {
    float soma = 0.0f;
    float resultado = 0.0f;

    for (int i = 0; i < N; i++) {
        soma += table[i][1];
    }

    float media = soma / N;

    for (int i = 0; i < N; i++) {
        float y = table[i][1];
        resultado += (y - media) * (y - media);
    }

    return sqrtf(resultado / (N - 1));
}


float ruido(float table[N][2], float w[DEGREE + 1]) {
    float soma = 0.0f;

    for (int i = 0; i < N; i++) {
        float x = table[i][0];
        float y = table[i][1];
        float prev = 0;
        for (int j=0; j<(DEGREE + 1); j++) {
            prev += w[j] * pow(x, j);
        }

        float e = y - prev;
        soma += e * e;
    }

    return sqrtf(soma / (N - DEGREE -1));
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
        // Remove \n ou \r\n do final
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
        table[i][1] = (float)atoi(reading)/100;

        printf("Aluno %d: math = %f, reading = %f\n",
               i + 1, table[i][0], table[i][1]);

        i++;
    }

    fclose(arquivo);

    printf("\nTotal de linhas carregadas: %d\n", i);

    printf("\n");
    float w[DEGREE + 1] ={0};
    float grads[DEGREE + 1]= {0};
    float argmin[DEGREE + 1] = {0};
    float Jmin = INFINITY;
    for (int i=0; i<EPOCHS; i++) {

        for (int j=0; j<DEGREE + 1; j++) {
            grads[j] = gradient(table, w, j);
        }

        for (int j=0; j<DEGREE + 1; j++) {
            w[j] = w[j] - ALPHA*grads[j];
        }

        if (J(table, w) < Jmin) {
            Jmin = J(table, w);
            for (int j=0; j<DEGREE + 1; j++) {
                argmin[j] = w[j];
            }
        }
    }

    printf("w = (%.6f", argmin[0]);
    for (int j = 1; j < (DEGREE + 1); j++) {
        printf(", %.6f", argmin[j]);
    }
    printf(")\n");
    printf("H(w) = %.6f", argmin[0]);

        for (int j = 1; j < (DEGREE + 1); j++) {
            if (j == 1) {
                printf(" + %.2fx", argmin[j]);
            } else {
                printf(" + %.2fx^%d", argmin[j], j);
            }
        }
    printf("\nJ(w) = %.6f", J(table, argmin));
    printf("\n");

    printf("\nRuido: %.2f \n", ruido(table, argmin));
    float perc = ruido(table, argmin)/dpy(table);
    printf("Ruido percentual: %.2f \n", perc*100);
    printf("Correlação: R²= %.2f \n",(1 - perc*perc)*100);

    float nota;
    float prev=0;
    while (1) {
        printf("Valor de previsão: X=");
        scanf("%f", &nota);
        for (int j=0; j<DEGREE + 1; j++) {
            prev += argmin[j]  *pow(nota, j);
        }
        printf("(%.2f, %.2f) \n", nota,  prev);
    }

}