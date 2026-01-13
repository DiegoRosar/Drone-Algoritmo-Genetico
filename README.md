Descrição

Este repositório contém uma implementação de um "simulador de rotas para drones", utilizando um Algoritmo Genético (GA) para tentar otimizar a ordem de visita a CEPs e as velocidades de voo, respeitando restrições operacionais definidas no enunciado do trabalho.

O código simula o voo de um drone a partir de um CEP base, visitando um conjunto de CEPs e retornando à base, levando em consideração:

-Consumo de bateria baseado na velocidade efetiva do voo
-Influência do vento na velocidade do drone
-Janelas de operação diária (horário permitido de voo)
-Limite máximo de dias de operação
-Tempo e custo de recarga
-Penalidades para pousos após determinado horário
-Penalizações numéricas para violações de restrições

A execução pode ocorrer em modo paralelo, com múltiplos agentes genéticos rodando simultaneamente por um tempo pré-definido, buscando minimizar uma função de custo (fitness) baseada em tempo, custo e penalidades.

Objetivo do código

O objetivo do projeto é avaliar a aplicabilidade de algoritmos genéticos para o problema de planejamento de rotas de drones sob múltiplas restrições reais, analisando:

-Capacidade do GA de melhorar soluções ao longo do tempo
-Impacto de penalidades suaves versus invalidação direta
-Limitações do GA frente a um espaço de busca muito grande
-Dificuldades práticas de convergência para soluções totalmente válidas

O projeto também inclui métricas e diagnósticos, como:

-Quantidade de CEPs visitados antes de falha
-CEPs mais problemáticos
-Evolução do melhor score ao longo do tempo

O que este código consiste

-Simulador físico-operacional de voo de drone
-Implementação completa de Algoritmo Genético (seleção, crossover, mutação, elitismo)
-Execução paralela com múltiplos agentes
-Geração de relatórios (`solution.csv`, estatísticas e diagnósticos)
-Ferramentas de análise para entender onde e por que as soluções falham

Observação importante

Apesar de longos tempos de execução e múltiplos agentes, o algoritmo nem sempre converge para soluções totalmente válidas, evidenciando as limitações do uso direto de algoritmos genéticos para este tipo de problema altamente restritivo e combinatorial.
Esse resultado faz parte da análise do trabalho e demonstra, na prática, os desafios de eficiência e escalabilidade desse tipo de abordagem.


Se quiser, no próximo passo eu posso:

* Ajustar o tom (mais técnico ou mais simples)
* Criar uma versão ainda mais curta
* Adaptar o texto para relatório acadêmico ou README.md
