# jax-qec

# jax-qec 🧠⚛️  
*A JAX-Based Toolkit for Discovering and Evaluating Quantum Error Correction Codes*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built with JAX](https://img.shields.io/badge/Built%20with-JAX-76B900.svg)](https://github.com/google/jax)

---

## ✨ Overview

**jax-qec** is a high-performance, open-source Python library for **discovering, simulating, and analyzing quantum error correction (QEC) codes**, powered by [JAX](https://github.com/google/jax). It is designed to enable fast experimentation with:

- ⚙️ Custom stabilizer codes
- 🧠 Learning-based code discovery (RL, Evolutionary Algorithms, XGBoost)
- 🎯 Noise-aware error correction optimization
- 🧪 Real-time evaluation of code distance and error resilience

This project is inspired by recent research in QEC code discovery using machine learning and reinforcement learning, but reimagined as a **modular, reusable, and educational toolkit** for the open-source quantum computing community.

---

## 🚀 Goals

- 📦 Build a modular JAX-based **Clifford simulator** for stabilizer codes
- 🔍 Implement multiple discovery engines (RL, EA, XGBoost pre-filtering)
- 📉 Support custom noise models for **hardware-aware QEC code design**
- 📊 Visualize performance metrics and track code families
- 🧪 Enable batch simulation and evaluation of QEC codes on GPU/TPU

---

## 🔨 Features

- ✅ Clifford simulator based on binary symplectic matrix algebra
- ✅ Efficient stabilizer updates using JAX's jit and vmap
- ✅ Pluggable code discovery agents:
  - Reinforcement Learning (PPO-based)
  - Evolutionary Algorithms
  - Gradient-free and ML-based filtering
- ✅ KL-based code evaluation with error model support
- ✅ Export of discovered codes for reuse and benchmarking

---

📚 Related Work
This project is inspired by and builds on ideas from:

Simultaneous Discovery of QEC Codes and Encoders with a Noise-Aware RL Agent
arXiv:2304.10392

Engineering Quantum Error Correction Codes Using Evolutionary Algorithms
arXiv:2301.11220

This repo is an independent implementation and extension with open-source accessibility, modularity, and broader usability as core goals.
