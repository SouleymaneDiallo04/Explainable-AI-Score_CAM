# Score-CAM — Explainability Without Gradients

**Mini-projet XAI · Décembre 2025**  
Souleymane Diallo 
Encadrant : Pr. Tawfik Masrour

---

## Pourquoi l'explicabilité des modèles de vision ?

Les réseaux de neurones convolutifs (CNN) atteignent aujourd'hui des performances remarquables en classification d'images, détection d'objets et analyse médicale. Mais leur nature de boîte noire pose un problème fondamental : **on ne sait pas ce que le modèle regarde pour prendre sa décision**.

Dans des domaines à fort enjeu — imagerie médicale, contrôle qualité industriel, conduite autonome — cette opacité est inacceptable. L'explicabilité (XAI) répond à ce besoin : rendre les décisions des modèles compréhensibles, auditables et dignes de confiance.

---

## Score-CAM : l'idée

Score-CAM est une méthode d'explicabilité **locale** pour les CNN. Elle produit une **carte de chaleur** (heatmap) qui met en évidence les régions d'une image ayant le plus influencé la prédiction du modèle.

Contrairement à Grad-CAM, Score-CAM **n'utilise aucun gradient**. Elle évalue l'importance de chaque région de manière directe et causale : en masquant successivement chaque zone et en mesurant l'impact réel sur le score de confiance du modèle.

```
Image → Cartes de caractéristiques → Masques normalisés → Score par masque → Heatmap finale
```

---

## Pipeline implémenté

Le notebook implémente Score-CAM de zéro sur un modèle **ResNet50 pré-entraîné** sur ImageNet :

1. **Extraction des cartes de caractéristiques** depuis la couche `layer4`
2. **Suréchantillonnage** bilinéaire de 7×7 à 224×224
3. **Normalisation Min-Max** pour transformer chaque carte en masque de transparence
4. **Calcul des scores** — une forward pass par masque, mesure de l'impact sur la classe cible
5. **Agrégation pondérée** par Softmax + ReLU pour produire la heatmap finale

La formulation mathématique est la suivante :

$$\alpha_k^c = f_c(A^k \odot X) - f_c(X_b)$$

$$L_{\text{Score-CAM}}^c = \text{ReLU}\left(\sum_{k=1}^{K} \alpha_k^c A^k\right)$$

---

## Résultats

Sur une image de chien (ImageNet), la heatmap Score-CAM se concentre précisément sur le museau et la queue — les régions les plus discriminantes — sans fuite d'importance vers le fond. Le résultat est plus propre et plus localisé que Grad-CAM sur ce cas d'usage.

---

## Forces et limites

| | Score-CAM |
|---|---|
| Stabilité | Pas de gradient → moins sensible au bruit |
| Localisation | Heatmaps plus nettes et précises |
| Interprétabilité | Importance = impact mesuré directement |
| Coût | Une forward pass par feature map → lent |
| Portée | CNN uniquement, pas adapté au texte ou au tabulaire |

Score-CAM ne doit pas être utilisé seul dans des contextes réglementés. Il gagne à être combiné avec des méthodes globales (SHAP, analyse d'erreurs) pour une validation robuste.

---

## Structure du notebook

```
XAI_Notebook.ipynb
├── Contexte & motivation
├── Intuition de la méthode
├── Formalisation mathématique
├── Implémentation pas à pas (ResNet50 + CIFAR-10)
│   ├── Extraction des feature maps
│   ├── Suréchantillonnage et normalisation
│   ├── Calcul des scores par masque
│   └── Agrégation et visualisation finale
├── Utilisation de pytorch-grad-cam (bibliothèque officielle)
├── Discussion critique : forces, limites, pièges
└── Conclusion & extensions
```

---

## Stack technique

`Python` `PyTorch` `torchvision` `pytorch-grad-cam` `NumPy` `Matplotlib`

---

## Références

- Wang et al., *Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks*, CVPR 2020
- Implémentation officielle : [github.com/haofanwang/Score-CAM](https://github.com/haofanwang/Score-CAM)
- TorchCAM : [github.com/frgfm/torch-cam](https://github.com/frgfm/torch-cam)
- Captum (Interpretability for PyTorch) : [captum.ai](https://captum.ai)
