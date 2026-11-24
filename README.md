# Machine Learning em C


Dataset: `StudentsPerformance.csv`

---

### Compilação

```bash
gcc main.c -o regressor -lm
```

A opção `-lm` é necessária porque o programa utiliza funções matemáticas

### Execução


Linux:

```bash
./regressor
```

Windows:

```bash
regressor.exe
```

---

## Campos

```text
gender, race/ethnicity, parental level of education, lunch,
test preparation course, math score, reading score, writing score
```

| Modelo | X | Y |
|---|---|---|
| Linear | `writing score` | `reading score` |
| Polinomial | `math score` | `reading score` |
| Logístico | `math score` | `gender` |



## Regressão linear

A regressão linear prevê `reading` a partir de `writing`.

```math
\hat{y} = w_0 + w_1x
```

- $w_0$ é o coeficiente linear
- $w_1$ é o coeficiente angular
- $x$ é a nota de `writing`
- $\hat{y}$ é a previsão de `reading`

### Função de custo

É utilizada para calcular a soma das distâncias entre os pontos e a reta

```math
J(w) = \frac{1}{2N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
```

### Gradiente descendente

Atualizado em cada iteração para gerar coeficientes com menor erro.

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

## Regressão polinomial

```math
\hat{y} = w_0 + w_1x + w_2x^2 + \cdots + w_dx^d
```

Ou:

```math
\hat{y} = \sum_{j=0}^{d} w_jx^j
```

### Grau 2

Para `DEGREE = 2`:

```math
\hat{y} = w_0 + w_1x + w_2x^2
```

### Grau 3

Para `DEGREE = 3`:

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

---

## Regressão logística

Neste projeto, o objetivo é prever se um aluno é `male` ou `female` a partir da nota de `math`.


```math
y =
\begin{cases}
1, & \text{se gender = male} \\
0, & \text{se gender = female}
\end{cases}
```

### Modelo

Primeiro, aplicamos os valores de x na formula da regressão polinomial

```math
z = w_0 + w_1x + w_2x^2 + \cdots + w_dx^d
```

Depois, colocamos eles na função logistica, para forçar um dominio entre 0 e 1
```math
\hat{y} = \sigma(z)
=
\frac{1}{1 + e^{-z}}
```

A saída $\hat{y}$ representa a probabilidade de o aluno ser `male`.

### Função de custo

Para calcular o custo, utilizamos a função de entropia cruzada binária:

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

O gradiente é exatamente igual ao da regressão polinomial:

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
--
## Perceptron

O perceptron não é uma regressão logistica simples, ele utiliza um neurônio de entrada, um neurônio oculto com ReLU e um neurônio de saída

```text
x -> ReLU -> sigmoid -> ŷ
```


```math
z_0 = w_{0,0} + w_{1,0}x
```

```math
a_0 = \mathrm{ReLU}(z_0)
```

```math
z_1 = w_{0,1} + w_{1,1}a_0
```

```math
\hat{y} = \sigma(z_1)
```

### Função ReLU

```math
\mathrm{ReLU}(x) =
\begin{cases}
0, & x < 0 \\
x, & x \geq 0
\end{cases}
```

A ReLU introduz não linearidade na rede.

### Derivada da ReLU

```math
\mathrm{ReLU}'(x) =
\begin{cases}
0, & x < 0 \\
1, & x > 0
\end{cases}
```

### Backpropagation

O erro da camada de saída:

```math
\delta_1 = \hat{y} - y
```

O gradiente dos pesos da camada de saída:

```math
\frac{\partial J}{\partial w_{0,1}} = \delta_1
```

```math
\frac{\partial J}{\partial w_{1,1}} = \delta_1a_0
```

O erro propagado para a camada oculta:

```math
\delta_0 = \delta_1w_{1,1}\mathrm{ReLU}'(z_0)
```

Os gradientes da camada oculta são:

```math
\frac{\partial J}{\partial w_{0,0}} = \delta_0
```

```math
\frac{\partial J}{\partial w_{1,0}} = \delta_0x
```

---

## 6. Hiperparâmetros

| Hiperparâmetro | Função | Efeito |
|---|---|---|
| `DEGREE` | Grau do polinômio | Quanto maior, mais complexo é o modelo |
| `EPOCHS` | Número de iterações | Quanto maior, mais oportunidades de convergência |
| `ALPHA` | Taxa de aprendizado | Controla o tamanho do passo de atualização |


### ALPHA

```math
w_j \leftarrow w_j - \alpha \frac{\partial J(w)}{\partial w_j}
```

Valores muito grandes podem causar divergência.

Valores muito pequenos podem deixar o treinamento lento.

| Escala de `x` | Alpha sugerido |
|---|---:|
| 0 a 1 | 0.01 a 0.1 |
| 0 a 10 | 0.001 a 0.01 |
| 0 a 100 | 0.00001 a 0.0001 |


---

## Métricas

### Ruído

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

Onde $\sigma_y$ é o desvio padrão de `reading`.

#### Coeficiente de determinação

```math
R^2 = 1 - \left(\frac{\text{ruído}}{\sigma_y}\right)^2
```

---

## Funções

| Função | Objetivo |
|---|---|
| `predict` | Previsão do modelo para x |
| `J` | Função de custo |
| `gradient` | Gradiente para um coeficiente(derivada parcial de J) |
| `dpy` | Calcula o desvio padrão de `reading` |
| `ruido` | Calcula o erro padrão residual |
