# PROJETO DE MACHINE LEARNING EM LINGUAGEM C

## REGRESSÃO LINEAR
Este projeto implementa, em C, uma **regressão linear simples** usando o algoritmo de **gradiente descendente**.


### Modelo

O modelo ajusta uma reta:

\[
H(x) = a + bx
\]

Onde:

- \(a\) é o intercepto
- \(b\) é o coeficiente angular
- \(x\) é a variável independente
- \(H(x)\) é o valor previsto

### Função de custo

A qualidade do modelo é medida pelo erro quadrático médio:

\[
J(a,b) = \frac{1}{N}\sum_{i=1}^{N}(H(x_i)-y_i)^2
\]

Ou, substituindo o modelo:

\[
J(a,b) = \frac{1}{N}\sum_{i=1}^{N}(a+bx_i-y_i)^2
\]

O objetivo do treinamento é encontrar os valores de \(a\) e \(b\) que minimizam \(J(a,b)\).

### Gradientes

A derivada parcial de \(J\) em relação ao intercepto \(a\) é:

\[
\frac{\partial J}{\partial a}
=
\frac{2}{N}\sum_{i=1}^{N}(a+bx_i-y_i)
\]

A derivada parcial em relação ao coeficiente angular \(b\) é:

\[
\frac{\partial J}{\partial b}
=
\frac{2}{N}\sum_{i=1}^{N}(a+bx_i-y_i)x_i
\]

### Atualização por gradiente descendente

Em cada iteração, os parâmetros são atualizados na direção oposta ao gradiente:

\[
a \leftarrow a - \alpha \frac{\partial J}{\partial a}
\]

\[
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
\]

Onde \(\alpha\) é a **taxa de aprendizado**.

### Ruído estimado

O resíduo de cada observação é a diferença entre o valor real e o valor previsto:

\[
e_i = y_i - H(x_i)
\]

O desvio padrão do ruído é estimado por:

\[
\sigma_{\text{ruído}}
=
\sqrt{\frac{\sum_{i=1}^{N} e_i^2}{N-2}}
\]

O denominador \(N-2\) é usado porque a regressão linear simples estima dois parâmetros: \(a\) e \(b\).

## Porcentagem de ruído

A porcentagem de ruído é calculada em relação ao desvio padrão dos valores reais de \(y\):

\[
\sigma_y =
\sqrt{\frac{\sum_{i=1}^{N}(y_i-\bar{y})^2}{N}}
\]

Onde:

\[
\bar{y} = \frac{1}{N}\sum_{i=1}^{N} y_i
\]

Assim, a porcentagem de ruído é:

\[
\text{ruído (\%)} =
100 \times \frac{\sigma_{\text{ruído}}}{\sigma_y}
\]

### Coeficiente de determinação

A relação aproximada entre a porcentagem de ruído e o \(R^2\) é:

\[
R^2 = 1 - \left(\frac{\sigma_{\text{ruído}}}{\sigma_y}\right)^2
\]

O \(R^2\) representa a proporção da variância de \(y\) explicada pelo modelo:

\[
R^2 = 1 - \frac{SSE}{SST}
\]

Onde:

\[
SSE = \sum_{i=1}^{N}(y_i-\hat{y}_i)^2
\]

\[
SST = \sum_{i=1}^{N}(y_i-\bar{y})^2
\]

### Dataset

O dataset utilizado é o **Students Performance in Exams**, disponível no Kaggle:

[Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?resource=download)

### Como compilar

```bash
gcc linear-regression.c -o linear-regression -lm
```

### Como executar

```bash
./linear-regression
```

No Windows, com MinGW:

```cmd
linear-regression.exe
```

## Como usar

Após o treinamento, informe um valor de \(x\) quando o programa solicitar. Ele retornará a previsão:

\[
\hat{y} = a + bx
\]

