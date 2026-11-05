# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <string.h>
#define MAX_LINHA 1024
# define N 1001

// H(x) = a + bx
float J(float table[N][2], float a, float b) {
    float soma = 0;
    for (int i = 0; i < 50; i++) {
        float x = table[i][0];
        float y = table[i][1];
        float previsao = a + b*x;
        soma += (previsao - y) * (previsao - y);
    }
    return soma/N*2;
}

float gradienteB(float table[N][2], float a, float b) {
    float soma = 0;
    for (int i = 0; i < 50; i++) {
        float x = table[i][0];
        float y = table[i][1];
        float previsao = a + b*x;
        soma += (previsao - y)*x;
    }
    return soma/N;
}

float gradienteA(float table[N][2], float a, float b) {
    float soma = 0;
    for (int i = 0; i < 50; i++) {
        float x = table[i][0];
        float y = table[i][1];
        float previsao = a + b*x;
        soma += (previsao - y);
    }
    return soma/N;
}

float ruido(float table[N][2], float a, float b) {
    float soma = 0;
    for (int i = 0; i < 50; i++) {
        float x = table[i][0];
        float y = table[i][1];
        float e = y - (a + b*x);
        soma += e*e;
    }
    return sqrtf(soma/(N - 2));

}

float dpy(float table[N][2], float a, float b) {
    float soma = 0;
    float resultado = 0;
    for (int i = 0; i < 50; i++) {
        soma += table[i][1];
    }
    float media = soma/N;
    for (int i = 0; i < 50; i++) {
        float y = table[i][1];
        resultado += (y - media) * (y - media);
    }
    return sqrtf(resultado/N);
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

        // table[i][0] = writing score
        // table[i][1] = reading score
        table[i][0] = (float)atoi(writing);
        table[i][1] = (float)atoi(reading);

        printf("Aluno %d: writing = %f, reading = %f\n",
               i + 1, table[i][0], table[i][1]);

        i++;
    }

    fclose(arquivo);

    printf("\nTotal de linhas carregadas: %d\n", i);

    printf("\n");
    float alpha = 0.001;
    float a = 0;
    float b = 0;
    float argmin[2];
    float Jmin = 9999999;
    for (int i=0; i<1000; i++) {
        a = a - alpha*gradienteA(table, a, b);
        b = b - alpha*gradienteB(table, a, b);

        if (J(table, a, b) < Jmin) {
            Jmin = J(table, a, b);
            argmin[0] = a;
            argmin[1] = b;
        }
    }
    printf("w=(%.2f ,%.2f)\n",argmin[0],argmin[1]);
    printf("J(w)=%.2f\n",Jmin);
    printf("H(w)=%.2f + %.2fx\n\n",argmin[0], argmin[1]);
    printf("Ruido: %f \n",ruido(table, argmin[0], argmin[1]));
    float noise_perc = ruido(table, argmin[0], argmin[1])/dpy(table, argmin[0], argmin[1]);
    printf("Porcentagem de ruido: %.2f \n",100*ruido(table, argmin[0], argmin[1])/dpy(table, argmin[0], argmin[1]));
    printf("Porcentagem de correlação: %.2f \n\n", (1 - noise_perc*noise_perc)*100);
    while (1) {
        float x;
        printf("Valor de teste:");
        scanf("%f", &x);
        printf("(%.2f, %.2f) \n", x, argmin[0] + argmin[1]*x);
    }
}