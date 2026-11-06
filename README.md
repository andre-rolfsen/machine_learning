# Regressores em C

Este projeto implementa três modelos de regressão/classificação em linguagem C, treinados por gradiente descendente:

- **Regressão linear**
- **Regressão polinomial**
- **Regressão logística**

Os modelos utilizam o arquivo `StudentsPerformance.csv` como base de dados.

---

## 1. Requisitos

Para compilar e executar os programas, você precisa de:

- Compilador C, como `gcc`
- Biblioteca matemática `math.h`
- Arquivo `StudentsPerformance.csv` no mesmo diretório do executável

### Compilando

```bash
gcc main.c -o regressor -lm
```

A opção `-lm` é necessária porque o programa utiliza funções matemáticas como `pow`, `sqrtf` e `exp`.

### Executando

```bash
./regressor
```

No Windows:

```bash
regressor.exe
```

---

## 2. Estrutura dos dados

O arquivo `StudentsPerformance.csv` deve conter as seguintes colunas:

```text
gender, race/ethnicity, parental level of education, lunch,
test preparation course, math score, reading score, writing score
```

| Modelo | Variável independente | Variável dependente |
|---|---|---|
| Linear | `math score` | `reading score` |
| Polinomial | `math score` | `reading score` |
| Logístico | `math score` | `gender` |

### Normalização

Para os modelos de regressão linear e polinomial, a nota de `math` é dividida por 100:

```math
x = \frac{\text{math}}{100}
```

Assim, a variável fica no intervalo:

```math
x \in [0, 1]
```

A normalização ajuda o gradiente descendente a convergir de forma mais estável.

---

## 3. Regressão linear

A regressão linear prevê `reading` a partir de `math`.

O modelo é:

```math
\hat{y} = w_0 + w_1x
```

Onde:

- `w_0` é o intercepto
- `w_1` é o coeficiente angular
- `x` é a nota de `math` dividida por 100
- `\hat{y}` é a previsão de `reading`

### Função de custo

```math
J(w) = \frac{1}{2N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
```

### Gradiente descendente

```math
w_j \leftarrow w_j - \alpha \frac{\partial J(w)}{\partial w_j}
```

O gradiente é:

```math
\frac{\partial J(w)}{\partial w_j}
=
\frac{1}{N}\sum_{i=1}^{N}
(\hat{y}_i - y_i)x_i^j
```

### Configuração recomendada

```c
#define DEGREE 1
#define EPOCHS 20000
#define ALPHA 0.01f
```

### Exemplo de saída

```text
w = (0.180360, 0.773727)
H(w) = 0.180360 + 0.77x
J(w) = 0.003545

Ruido: 0.08
Ruido percentual: 57.12
Correlação: R² = 67.37%
```

---

## 4. Regressão polinomial

A regressão polinomial generaliza a regressão linear.

Para grau `d`, o modelo é:

```math
\hat{y} = w_0 + w_1x + w_2x^2 + \cdots + w_dx^d
```

Em forma compacta:

```math
\hat{y} = \sum_{j=0}^{d} w_jx^j
```

### Grau 2

Com `DEGREE = 2`:

```math
\hat{y} = w_0 + w_1x + w_2x^2
```

### Grau 3

Com `DEGREE = 3`:

```math
\hat{y} = w_0 + w_1x + w_2x^2 + w_3x^3
```

### Função de custo

```math
J(w) = \frac{1}{2N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
```

### Gradiente

```math
\frac{\partial J(w)}{\partial w_j}
=
\frac{1}{N}\sum_{i=1}^{N}
(\hat{y}_i - y_i)x_i^j
```

### Configuração recomendada

```c
#define DEGREE 2
#define EPOCHS 20000
#define ALPHA 0.01f
```

### Resultados obtidos

| Modelo | `J(w)` | Ruído | `R²` |
|---|---:|---:|---:|
| Linear | 0.003545 | 0.08 | 67.37% |
| Quadrático | 0.003633 | 0.09 | 66.53% |
| Cúbico | 0.003678 | 0.09 | 66.08% |

Neste dataset, o modelo linear apresentou o melhor resultado.

---

## 5. Regressão logística

A regressão logística é usada para classificação binária.

Neste projeto, o objetivo é prever se um aluno é `male` ou `female` a partir da nota de `math`.

A variável dependente é codificada como:

```math
y =
\begin{cases}
1, & \text{se gender = male} \\
0, & \text{se gender = female}
\end{cases}
```

### Modelo

Primeiro, calcula-se o valor linear:

```math
z = w_0 + w_1x + w_2x^2 + \cdots + w_dx^d
```

Depois, aplica-se a função sigmoide:

```math
\hat{y} = \sigma(z)
=
\frac{1}{1 + e^{-z}}
```

A saída `\hat{y}` representa a probabilidade de o aluno ser `male`.

### Função de custo

A regressão logística utiliza a entropia cruzada binária:

```math
J(w)
=
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i\log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right]
```

### Gradiente

```math
\frac{\partial J(w)}{\partial w_j}
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i - y_i)x_i^j
```

### Regra de decisão

```math
\hat{y} > 0.5 \Rightarrow \text{male}
```

```math
\hat{y} \leq 0.5 \Rightarrow \text{female}
```

### Configuração recomendada

```c
#define DEGREE 1
#define EPOCHS 5000
#define ALPHA 0.1f
```

### Previsão

Para prever um novo aluno, informe a nota de `math` na escala original, de 0 a 100.

O programa divide o valor por 100 antes de aplicar o modelo:

```math
x = \frac{\text{nota}}{100}
```

Em seguida, calcula:

```math
P(\text{male}) = \frac{1}{1 + e^{-z}}
```

---

## 6. Hiperparâmetros

| Hiperparâmetro | Função | Efeito |
|---|---|---|
| `DEGREE` | Define o grau do polinômio | Quanto maior, mais complexo é o modelo |
| `EPOCHS` | Número de iterações do gradiente descendente | Quanto maior, mais oportunidades de convergência |
| `ALPHA` | Taxa de aprendizado | Controla o tamanho do passo de atualização |

### DEGREE

Para regressão linear:

```c
#define DEGREE 1
```

Para regressão polinomial:

```c
#define DEGREE 2
```

Para regressão logística:

```c
#define DEGREE 1
```

### EPOCHS

```c
#define EPOCHS 20000
```

Valores maiores podem melhorar a convergência, mas aumentam o tempo de execução.

### ALPHA

```math
w_j \leftarrow w_j - \alpha \frac{\partial J(w)}{\partial w_j}
```

Valores muito grandes podem causar divergência.

Valores muito pequenos podem deixar o treinamento lento.

### Sugestões de alpha

| Escala de `x` | Alpha sugerido |
|---|---:|
| 0 a 1 | 0.01 a 0.1 |
| 0 a 10 | 0.001 a 0.01 |
| 0 a 100 | 0.00001 a 0.0001 |

Como este projeto normaliza `math` dividindo por 100, `ALPHA = 0.01f` funciona bem para regressão linear e polinomial.

---

## 7. Métricas

### Regressão linear e polinomial

#### Ruído

```math
\text{ruído}
=
\sqrt{
\frac{1}{N-d-1}
\sum_{i=1}^{N}
(y_i - \hat{y}_i)^2
}
```

Onde `d` é o grau do polinômio.

#### Ruído percentual

```math
\text{ruído percentual}
=
\frac{\text{ruído}}{\sigma_y} \times 100
```

Onde `\sigma_y` é o desvio padrão de `reading`.

#### Coeficiente de determinação

```math
R^2 = 1 - \left(\frac{\text{ruído}}{\sigma_y}\right)^2
```

Em porcentagem:

```math
R^2_{\%} = R^2 \times 100
```

### Regressão logística

A métrica adequada é a acurácia:

```math
\text{acurácia}
=
\frac{\text{número de acertos}}{N}
```

Em porcentagem:

```math
\text{acurácia}_{\%}
=
\frac{\text{número de acertos}}{N} \times 100
```

---

## 8. Estrutura do código

| Função | Objetivo |
|---|---|
| `predict` | Calcula a previsão do modelo |
| `J` | Calcula a função de custo |
| `gradient` | Calcula o gradiente para um coeficiente |
| `dpy` | Calcula o desvio padrão de `reading` |
| `ruido` | Calcula o erro padrão residual |
| `main` | Carrega os dados, treina o modelo e faz previsões |

### Fluxo do programa

1. Abre o arquivo CSV
2. Lê os dados
3. Normaliza a variável `math`
4. Treina o modelo por gradiente descendente
5. Guarda os melhores coeficientes encontrados
6. Imprime os coeficientes e métricas
7. Permite previsões interativas

---
