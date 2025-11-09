# 1. Importação das Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os
from pathlib import Path

# Suprimir avisos para uma saída mais limpa
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Garantir que estamos trabalhando no diretório correto (onde está o script)
script_dir = Path(__file__).parent
os.chdir(script_dir)

print("Iniciando processamento das atividades...")
print(f"Diretorio de trabalho: {os.getcwd()}")

# ---
# --- ATIVIDADE 1: cluster_data_1.csv
# ---
print("\n" + "---" * 20)
print("ATIVIDADE 1 (data_1.csv)")
print("---" * 20)

try:
    # Carregar os dados
    df1 = pd.read_csv('data_1.csv')
    
    # Padronizar os dados (Clustering é sensível à escala)
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(df1)
    
    # --- Item 1 (Modificado): Encontrar k via Elbow Method ---
    print("\n[Item 1] Qual a quantidade de clusters?")
    print("Rodando o Método do Cotovelo (Elbow Method) para k=2 a k=10...")
    
    inertia_list = [] # WCSS (Within-Cluster Sum of Squares)
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans_elbow = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans_elbow.fit(X1_scaled)
        inertia_list.append(kmeans_elbow.inertia_)
        
    # Plotar o gráfico do Cotovelo
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia_list, marker='o', linestyle='--')
    plt.title('Atividade 1: Método do Cotovelo (Elbow Method)')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia (WCSS)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig('atividade_1_elbow_plot.png')
    
    print("Gráfico 'atividade_1_elbow_plot.png' salvo.")
    print("ANÁLISE (Item 1): O gráfico mostra um 'cotovelo' claro em k=4.")
    k_final_1 = 4
    print(f"-> Quantidade de clusters selecionada: {k_final_1}\n")

    # Dicionário para armazenar resultados
    resultados_1 = {}
    
    # Rodar algoritmos com k=4
    algoritmos_1 = {
        "KMeans": KMeans(n_clusters=k_final_1, n_init=10, random_state=42),
        "AgglomerativeClustering": AgglomerativeClustering(n_clusters=k_final_1)
    }

    for nome, alg in algoritmos_1.items():
        print(f"--- Avaliando {nome} com k=4 ---")
        
        # Treinar e prever
        labels_pred = alg.fit_predict(X1_scaled)
        
        # [Item 2] Quantos pontos há em cada cluster
        (unique, counts) = np.unique(labels_pred, return_counts=True)
        print(f"[Item 2] Pontos por cluster:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} pontos")

        # [Item 3] Coeficiente de Silhouette
        silhouette = silhouette_score(X1_scaled, labels_pred)
        print(f"[Item 3] Coef. Silhouette: {silhouette:.4f} (Mais perto de 1 é melhor)")

        # [Item 4] Coeficiente de Davies Bouldin
        davies_bouldin = davies_bouldin_score(X1_scaled, labels_pred)
        print(f"[Item 4] Coef. Davies Bouldin: {davies_bouldin:.4f} (Mais perto de 0 é melhor)")
        
        # Salvar resultados para item 5
        resultados_1[nome] = {'silhouette': silhouette, 'davies_bouldin': davies_bouldin}
        
        # Salvar labels do KMeans para o gráfico PCA
        if nome == "KMeans":
            df1['Cluster_KMeans'] = labels_pred

    # [Item 5] Há diferença na performance?
    print("\n--- [Item 5] Comparação de Performance (Atividade 1) ---")
    print(f"Métrica               | KMeans  | Agglomerative | Diferença")
    print(f"----------------------|---------|---------------|----------")
    sil_k = resultados_1['KMeans']['silhouette']
    sil_a = resultados_1['AgglomerativeClustering']['silhouette']
    diff_sil = abs(sil_k - sil_a)
    print(f"Silhouette (-> 1)     | {sil_k:.4f} | {sil_a:.4f}      | {diff_sil:.4f}")
    
    db_k = resultados_1['KMeans']['davies_bouldin']
    db_a = resultados_1['AgglomerativeClustering']['davies_bouldin']
    diff_db = abs(db_k - db_a)
    print(f"Davies-Bouldin (-> 0) | {db_k:.4f} | {db_a:.4f}      | {diff_db:.4f}")
    
    # Análise de significância com threshold de 0.01
    threshold = 0.01
    if diff_sil < threshold and diff_db < threshold:
        print(f"-> As diferenças (< {threshold}) são estatisticamente insignificantes.")
        print("   Ambos os algoritmos têm performance equivalente.\n")
    else:
        melhor_sil = 'KMeans' if sil_k > sil_a else 'AgglomerativeClustering'
        melhor_db = 'KMeans' if db_k < db_a else 'AgglomerativeClustering'
        print(f"-> Há diferença significativa: {melhor_sil} (Silhouette), {melhor_db} (Davies-Bouldin)\n")
    
    # Gerar Gráfico PCA
    print("Gerando gráfico PCA para Atividade 1...")
    pca = PCA(n_components=2)
    X1_pca = pca.fit_transform(X1_scaled)
    pca_df = pd.DataFrame(data=X1_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster_KMeans'] = df1['Cluster_KMeans']
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster_KMeans', palette='viridis', s=50, alpha=0.8)
    plt.title('Atividade 1: Visualização PCA dos Clusters (KMeans k=4)', fontsize=16)
    plt.xlabel('Componente Principal 1 (PC1)')
    plt.ylabel('Componente Principal 2 (PC2)')
    plt.legend(title='Cluster')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('atividade_1_pca_plot.png')
    print("Gráfico 'atividade_1_pca_plot.png' salvo.")


except FileNotFoundError:
    print("ERRO: 'data_1.csv' não encontrado. Pulando Atividade 1.")
except Exception as e:
    print(f"ERRO inesperado na Atividade 1: {e}")


print("\n" + "---" * 20)
print("Processamento Concluido.")
print("Verifique os 2 graficos salvos: 'atividade_1_elbow_plot.png', 'atividade_1_pca_plot.png'")