# 🧬 Machine Learning for CRISPR Efficiency Enhancement: Prime Editing Focus

## 🏆 Project Overview
This project aims to develop an ML model that predicts CRISPR Prime Editing efficiency, pushing the frontiers of genetic editing by using advanced methods. Transformer architectures and Bayesian optimization are employed to model DNA/RNA sequences, improving upon existing GRU and CNN models. This project aims to develop an advanced machine learning model to predict the efficiency of CRISPR DNA editing, with a focus on Prime Editor technology. CRISPR enables precise DNA edits and has significant implications for gene therapy. We will explore transformer-based architectures and Bayesian optimization to enhance the predictive performance over existing CNN and GRU models. The dataset includes over 300,000 data points with features such as DNA target sequences, RNA guides, CRISPR efficiency scores, GC content, and melting points.


## 👥 Team Roles and Contributions

- **Project Lead & Data Pre-processing**  
  Focuses on data pre-processing, including the development of methods to accurately capture the structural properties of DNA and RNA sequences.

- **Model Development & Optimization**  
  Responsible for model development and optimization using advanced techniques tailored for DNA and RNA sequences.

- **Evaluation & Visualization**  
  Oversees the evaluation of the model using appropriate metrics and develops visualizations to represent the model’s predictive capabilities.


## 🔧 ML Related Topics

- Application of transformer architectures for biological sequence data analysis.
- Utilization of Bayesian optimization for efficient hyperparameter tuning.
- Advanced data pre-processing techniques for biological sequences.
- Evaluation metrics such as sensitivity, specificity, precision, recall, F1-score, AUC-ROC, and AUC-PR.

## 📊 Dataset Description

The dataset consists of 300,000+ data points with DNA sequences, RNA guides, and CRISPR efficiency scores. Additional features such as GC content and melting points were leveraged.

- **Dataset links**:  
  [NCBI](https://www.ncbi.nlm.nih.gov/sra/SRX18661809[accn])  
  [DeepPrime Dataset](https://github.com/yumin-c/DeepPrime/blob/master/data/DeepPrime_dataset_final_Feat8.csv)



## 🎯 Conclusion

This project pushes the boundaries of CRISPR-Cas9 research by enhancing the ability to predict prime editing efficiency. With transformer-based architectures and improved data preprocessing, we aim to improve experimental efficiency and accuracy for genetic editing, empowering biologists to refine CRISPR experiments.

Developed ML model using transformer architectures to predict DNA CRISPR Prime Editing efficiency. 
-	Applied one-hot encoding and embedding layers to efficiently capture key biological sequence features like GC content and melting points for better DNA/RNA structure representation.
-	Used Optuna for Bayesian optimization, tuning batch size, learning rate, and node counts. Despite memory limits, achieved MSE losses of 5.3 and 6.2, showing room for further improvement.
-	Led evaluation and visualization, using metrics such as sensitivity, precision, F1-score, and AUC-ROC, and developed clear visualizations to represent the model’s performance.


## 🔗 References
- Prediction of efficiencies for diverse prime editing systems in multiple cell types, *Cell*. 2023 May
