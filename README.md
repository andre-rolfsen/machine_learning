# PROJETO DE MACHINE LEARNING EM C

## Regressão Linear em C
(linear-regression.c)
Este projeto implementa, em C, uma **regressão linear simples** utilizando o algoritmo de **gradiente descendente**.

O objetivo é prever a nota de **leitura** (`reading score`) a partir da nota de **escrita** (`writing score`) de estudantes

## Dataset

O dataset utilizado é o **Students Performance in Exams**, disponível no Kaggle.

**Link para download:**

[Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?resource=download)

Após baixar o arquivo, coloque-o na mesma pasta do código-fonte e renomeie-o para:

```text
StudentsPerformance.csv
```

## Estrutura do CSV

O programa espera um arquivo CSV com o seguinte cabeçalho:

```csv
"gender","race/ethnicity","parental level of education","lunch","test preparation course","math score","reading score","writing score"
```

O programa utiliza:

- `writing score` como variável independente $x$
- `reading score` como variável dependente $y$

## Modelo

O modelo ajusta uma reta:

$$
H(x) = a + bx
$$

Onde:

- $a$ é o coeficiente linear
- $b$ é o coeficiente angular
- $x$ é a nota de escrita
- $H(x)$ é a previsão da nota de leitura

## Função de custo

A qualidade do modelo é medida pelo erro quadrático médio:

$$
J(a,b) = \frac{1}{2N}\sum_{i=1}^{N}(H(x_i)-y_i)^2
$$

O objetivo do treinamento é encontrar os valores de $a$ e $b$ que minimizam $J(a,b)$.

## Gradiente descendente

O gradiente descendente atualiza os parâmetros na direção oposta ao gradiente da função de custo.

Para o erro quadrático médio:

```math
J(a,b) = \frac{1}{2N}\sum_{i=1}^{N}(a + bx_i - y_i)^2
```

A derivada parcial em relação ao coeficiente linear é:

```math
\frac{\partial J}{\partial a} = \frac{1}{N}\sum_{i=1}^{N}(a + bx_i - y_i)
```

A derivada parcial em relação ao coeficiente angular é:

```math
\frac{\partial J}{\partial b} = \frac{1}{N}\sum_{i=1}^{N}(a + bx_i - y_i)x_i
```

Os dois gradientes devem ser calculados usando os mesmos valores atuais dos parâmetros, antes de realizar as atualizações:

```math
a_{\mathrm{novo}} = a - \alpha \frac{\partial J}{\partial a}
```

```math
b_{\mathrm{novo}} = b - \alpha \frac{\partial J}{\partial b}
```

O parâmetro de taxa de aprendizado controla o tamanho de cada atualização.

## Ruído estimado

O resíduo de cada observação é a diferença entre o valor real e o valor previsto:

$$
e_i = y_i - H(x_i)
$$

## Desvio padrão residual

```math
s_e = \sqrt{\frac{\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}{N-2}}
```

O denominador $N-2$ é utilizado porque a regressão linear simples estima dois parâmetros: $a$ e $b$.

## Porcentagem de ruído

A porcentagem de ruído é calculada em relação ao desvio padrão dos valores reais de $y$.

Primeiro, calcula-se a média:

$$
\bar{y} = \frac{1}{N}\sum_{i=1}^{N} y_i
$$

Depois, o desvio padrão:

$$
\sigma_y =
\sqrt{\frac{\sum_{i=1}^{N}(y_i-\bar{y})^2}{N}}
$$

Por fim, a porcentagem de ruído é:

$$
\text{ruído \%} =
100 \times \frac{\sigma_{\text{ruído}}}{\sigma_y}
$$

## Coeficiente de determinação

A relação aproximada entre a porcentagem de ruído e o $R^2$ é:

$$
R^2 = 1 - \left(\frac{\sigma_{\text{ruído}}}{\sigma_y}\right)^2
$$


## Pré-requisitos

É necessário ter o compilador GCC instalado.

### Linux

```bash
sudo apt update
sudo apt install build-essential
```

### Windows

Você pode utilizar:

- WSL com Ubuntu
- MinGW-w64
- MSYS2

## Como compilar

Na pasta do projeto, execute:

```bash
gcc linear-regression.c -o linear-regression -lm
```

O parâmetro `-lm` é necessário porque o programa utiliza funções matemáticas da biblioteca `libm`, como `sqrtf()`.

## Como executar

```bash
./linear-regression
```

No Windows, com MinGW:

```cmd
linear-regression.exe
```

## Como usar

Após o treinamento, o programa exibirá:

- Os coeficientes aprendidos $a$ e $b$
- O custo final $J(a,b)$
- A equação da reta ajustada
- O desvio padrão do ruído
- A porcentagem de ruído

Depois, informe uma nota de escrita para receber a previsão da nota de leitura.
