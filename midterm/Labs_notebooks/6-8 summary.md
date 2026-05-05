### 1. Huawei Ascend Computing Platform (6.1)

**What I learned:**

* The Ascend processor uses the  **Da Vinci architecture** , which is specifically designed for AI computing with **99.2% of its core dedicated to Cube units** for matrix multiplication. A single Cube unit can perform 4096 FP16 operations per clock cycle (16x16 matrix multiply).
* The AI Core contains three execution units: **Cube** (matrix), **Vector** (vector), and **Scalar** (scalar/program control). The storage system includes input/output buffers and registers to reduce bus traffic.
* AI processors are classified into **training** (high throughput, large memory, e.g., Atlas 900) and **inference** (low power, low latency, e.g., Atlas 200/300 series).
* The **Atlas computing platform**
  scales from edge (Atlas 200, 500) to data center (Atlas 800
  training/inference, Atlas 900 cluster). For instance, Atlas 900 can
  deliver **20.4 PFLOPS FP16** with ultra‑high energy efficiency.
* Huawei uses Ascend in real‑world projects, like the  **Peng Cheng Cloud Brain II** , reducing TCO by 9.3% with the same computing power, and the  **Traffic Brain** , improving violation detection by 10×.

**Why it was interesting/useful:**

* I was impressed by the  **Cube unit** ’s
  massive parallelism (up to 4096 operations per cycle) and how tightly
  the hardware is optimized for tensor operations. The comparison with
  GPUs (only a small fraction of Tensor Cores) clearly showed why
  dedicated NPUs achieve higher efficiency.
* The **hardware‑software co‑design**
  (CANN, DVPP, runtime manager) demonstrates how a full stack enables
  high performance. The real‑world case of reducing power and space by
  60–80% in data centers makes the technology tangible.

---

### 2. Huawei Cloud EI Platform (6.2)

**What I learned:**

* **ModelArts**
  is a one‑stop AI development platform that covers the entire ML
  lifecycle: data labeling, training (ExeML for beginners), deployment,
  and MLOps. The **ExeML engine** allows people without programming skills to build models in three steps.
* Advanced services include **OptVerse AI Solver** for operations optimization, **federated learning** to break data silos, and **ModelBox** for device‑edge‑cloud joint inference, improving E2E performance by up to 3×.
* General AI capabilities: **vision services** (100+ table types, face search >95.5%), **NLP** (industry‑leading accuracy in NER, sentiment analysis, summarization), **knowledge graph** (full‑lifecycle management), and **Conversational Bot Service** (handled workload of 179 human agents annually at Huawei).
* The platform also offers third‑party integration and an **AI Gallery** bridging supply and demand.

**Why it was interesting/useful:**

* I really liked **ExeML** because it shows how AI can be democratised—users only need to upload and label data, train, then publish a model. The **federated learning** capability is crucial for industries like healthcare and finance that cannot share raw data.
* The **CBS bot**
  case study (serving 350,000+ Huawei staff, 65% manual substitution
  rate) proved the immediate ROI of cloud AI services. I can see myself
  using such cloud APIs for rapid prototyping.

---

### 3. Huawei Device AI Platforms (6.3)

**What I learned:**

* **HarmonyOS** is a distributed OS enabling seamless collaboration across devices.
* **HMS Core** offers capabilities in seven domains to build an intelligent app ecosystem.
* **ML Kit**
  provides a wide range of AI APIs (text, language, image, face/body)
  that work across Android, iOS, and HarmonyOS, with on‑device processing
  for data security.
* **HiAI** is a three‑layer open platform: **Foundation** opens NPU acceleration (150+ operators), **Engine** delivers rich AI capabilities out of the box, and **Service** connects users to intelligent services.
* **MindSpore Lite**
  is an ultra‑fast, lightweight AI engine that supports CPU/GPU/NPU,
  model compression, and can run on everything from phones to IoT devices.
  It is compatible with MindSpore, TensorFlow Lite, Caffe, and ONNX.

**Why it was interesting/useful:**

* The concept of **on‑device AI**
  (ML Kit, HiAI) is fascinating because it ensures privacy (data never
  leaves the device) and reduces latency. As a developer, I can use the  **HiAI Foundation** ’s operators to get NPU acceleration without deep hardware knowledge.
* **MindSpore Lite** ’s
  claim of “all‑scenario support” and fast deployment for extreme
  environments makes it very practical for edge AI projects. The fact that
  it can quantize and compress models for tiny devices is something I
  want to explore further.

---

### 4. Cutting‑edge AI Applications (7)

**What I learned:**

* **Reinforcement learning (RL):**
  agents learn optimal policies through interaction with the environment.
  Algorithms can be direct (policy optimization) or indirect
  (value‑based).
* **GANs:**
  a generator and a discriminator compete, eventually creating realistic
  fake data. Applications include image generation, text‑to‑image,
  super‑resolution, and speech enhancement.
* **Knowledge graphs:**
  structured semantic networks of entities, properties, and
  relationships. Construction involves extraction, fusion, and storage.
  Applications: search engines, Q&A bots, recommendation systems, and
  domain‑specific solutions (e.g., oil & gas).
* **Intelligent driving:**
  history from Carnegie Mellon’s Navlab (1995, 98.1% autonomous across
  the US) to DARPA challenges (2004–2007) that fostered modern
  self‑driving cars. Today’s autonomous driving cloud service offers a
  one‑stop platform with data, training, and simulation services.

**Why it was interesting/useful:**

* The **GAN explanation**
  made the adversarial concept very clear: “the generator creates fakes,
  the discriminator tries to spot them.” I found the practical uses
  (resolution enhancement, text‑to‑image) inspiring.
* The **historical timeline of intelligent driving**
  was eye‑opening—it highlighted that many foundational ideas date back
  to the 1990s. The DARPA challenges were a brilliant catalyst for the
  industry. The modular autonomous driving cloud platform
  (data‑training‑simulation) gave me insight into how one might build such
  a system today.

---

### 5. Quantum Computing and Machine Learning (8)

**What I learned:**

* **Quantum mechanics basics:** superposition, entanglement, and uncertainty. A **qubit** can be in a superposition of |0⟩ and |1⟩, enabling exponential state space.
* **Key algorithms:**
  * *Deutsch algorithm* solves a simple problem in one query instead of two, demonstrating quantum advantage.
  * *Shor* factorizes integers in polynomial time, threatening RSA.
  * *Grover* searches unsorted databases with √N complexity.
  * *HHL* solves linear systems exponentially faster than classical methods.
* **Quantum machine learning (QML)**
  combines classical data/algorithms with quantum resources. Variational
  algorithms (e.g., VQE for chemistry) use a hybrid classical‑quantum loop
  to find ground state energies. QML promises acceleration for tasks like
  PCA, SVM, and reinforcement learning.
* **MindSpore Quantum**
  is an open‑source framework that integrates with MindSpore, offering
  high‑performance simulators (up to 30+ qubits), rich tutorials, and
  seamless cloud deployment. I got hands‑on with a code snippet that
  builds a circuit, computes expectation values, and gradients.

**Why it was interesting/useful:**

* The idea of **exponential parallelism**
  through superposition is mind‑blowing—the Deutsch algorithm showing we
  can solve a problem with one function call instead of two illustrated a
  tangible, though small, quantum advantage.
* **MindSpore Quantum** ’s
  simplicity (PIP install, visualization, built‑in libraries) makes it
  approachable even for an ML student like me. The example of calculating
  expected values and gradients in just a few lines of code makes me
  excited to try quantum machine learning experiments. The VQE application
  to chemistry shows how quantum computing can address real scientific
  challenges today.
