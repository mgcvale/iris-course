# Fronteiras de decisão

Visualizador didático para desenhar pontos e comparar as fronteiras aprendidas
por regressão logística, SVM, KNN e algoritmo genético.

## SVM

A interface permite escolher:

- kernel linear, RBF, polinomial ou sigmoide;
- valor de `C`;
- `gamma` como `scale`, `auto` ou um número;
- grau do kernel polinomial.

## Como rodar com um único servidor

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e Node.js.

```bash
cd front
npm install
npm run build

cd ../api
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Abra <http://localhost:8000>. O FastAPI serve a API, o WebSocket e os arquivos
compilados de `front/dist` na mesma porta. Node.js é necessário somente para
gerar o front; em execução há apenas o processo do `uvicorn`.

## Desenvolvimento do front

Se quiser hot reload, rode a API normalmente e, em outro terminal:

```bash
cd front
npm run dev -- --port 5174
```

O Vite encaminha `/api` e `/ws` para a API na porta 8000.

## Estrutura

```text
api/    FastAPI, datasets e modelos
front/  HTML, CSS e JavaScript compilados pelo Vite
docs/   imagens históricas do projeto
```
