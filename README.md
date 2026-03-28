# Big Data Application  

## Data Crawling Pipeline - Implicit Feedback Recommendation System

## 📌 Overview
This project focuses on building a **data crawling and preprocessing pipeline** for a **Recommendation System with Implicit Feedback**.  
Instead of explicit ratings (e.g. 1–5 stars), the system leverages **user behavior signals**

Starring is a natural and lightweight action that reflects a user's interest or approval of a repository.  
Unlike explicit rating systems, GitHub does not require users to provide numerical scores; instead, actions such as starring are performed voluntarily and organically during normal usage.

Moreover, starring behavior has several advantages:
- It indicates **positive user preference** without requiring additional effort from the user
- It is **widely adopted and consistently available** across repositories
- It is less noisy than transient signals (e.g., page views) and more stable over time

Therefore, repository starring serves as a reliable and scalable form of implicit feedback for modeling user–item interactions in a GitHub-based recommendation system.

The project is designed for **big data applications**, emphasizing:
- Large-scale data collection via APIs
- Implicit feedback modeling
- Scalable data preprocessing for recommendation algorithms

---

## 🎯 Problem Definition
Traditional recommendation systems rely on explicit feedback, which is often:
- Sparse
- Biased
- Hard to collect

This project addresses these issues by formulating the recommendation task as an **implicit feedback problem**, where:
- Positive interactions imply preference
- Missing interactions are treated as unobserved (not negative)

---

## 🔗 Link

Data collect: https://drive.google.com/drive/folders/1iHCPeFCWLCx9O7nBxBL8lX9jAaMhnF5v?hl=vi
