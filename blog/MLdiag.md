---


---

<h1 id="mldiag">MLDiag</h1>
<p>an effective machine learing model diagnosing tool</p>
<h2 id="introduction">Introduction</h2>
<p>Testing machine learning models is usually limited to some actions, mainly:</p>
<ul>
<li>algorithmic performances (metrics) of the model over a validation  set (accuracy, precision, recall, etc.),</li>
<li>computational performances of the model (speed, memory, etc.)</li>
<li>examples where the model was most confidently incorrect,</li>
</ul>
<p>Decision made to promote a model into production is usually based on theses  observations:</p>
<ul>
<li>is the current model offering an improvement over the existing version when evaluated on the same dataset.</li>
<li>is the current model reaching the business required performance.</li>
</ul>
<p>The model that fulfill this checklist is saved, along with hyperparamters and dataset that were used to validate the model, using MLFlow for instance. However, this traditional model checking lacks many important features:</p>
<ul>
<li><strong>where the model  does usually fail, and how to fix it?</strong></li>
<li><strong>does the model generalize well on some variations over the evaluation set?</strong></li>
<li><strong>how to track (and prevent) behavioral regressions for specific failure modes that had been previously addressed?</strong> Indeed, you might improve the overall evaluation metric but introduce a regression on a critical subset of data. Or you could unknowingly add a gender bias to the model through the inclusion of a new dataset during training.</li>
<li><strong>how can we define the best directions for model future improvement</strong> researcher usually look at a small set of incorrect predicted data to understand model failure. Such approach is biased since small samples  are likely unrepresentative of the true error distribution. A more precise, reproducible, scalable, and testable procedure is required.</li>
</ul>
<p>The checklist could be larger, but work is usually made manually.</p>
<p>MLDiag is a simple, yet effective framework, to handle model evaluation automatically. It provides at the end of the evaluation a comprehensive report on model robustness to adversarial attacks and simple tracks to improve its performances.</p>
<h2 id="related-work">Related work</h2>
<p>Recent works have tackled the model evaluation step.</p>
<p>Odena et al. introduced in this <a href="http://proceedings.mlr.press/v97/odena19a/odena19a.pdf">paper</a> in MLR 2019  testing techniques for neural networks that can discover errors occurring only for rare inputs., called <em>Coverage-Guided Fuzzing (CGF)</em> methods, which are random mutations of inputs guided by a coverage metric toward the goal of satisfying user-specified constraints.<br>
<code>TODO: read paper</code></p>
<p>Ribeiro et al. introduced in this <a href="https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf">paper</a> in ACL 2020 a task-agnostic methodology for testing NLP models, called <em>CheckList</em>.<br>
<code>TODO: read paper</code></p>
<p>Jordan proposed in his <a href="https://www.jeremyjordan.me/testing-ml/?utm_campaign=Data_Elixir&amp;utm_source=Data_Elixir_300">blog</a> in 2020 two behavior testing recipes to address the model testing step:</p>
<ul>
<li><em>Pre-train tests</em> that allow us to identify some bugs early on and short-circuit a training job.</li>
<li><em>Post-train tests</em> use the trained model artifact to inspect behaviors for a variety of important scenarios that we define.</li>
</ul>
<p>Naike et al. proposed in this <a href="https://arxiv.org/pdf/1806.00692.pdf">paper</a> in ICCL 2018 an evaluation methodology consisting of automatically constructed “stress tests” that allow to examine whether systems have the ability to make real inferential decisions. An evaluation of six sentence-encoder models on these stress tests reveals strengths and weaknesses of these models with respect to challenging linguistic phenomena.<br>
<code>TODO: read paper</code></p>
<p>Belinkov and Binsk find in this <a href="https://arxiv.org/pdf/1711.02173.pdf">study</a> in ICLR 2018 that state-of-the-art models fail to translate even moderately noisy texts that humans have no trouble comprehending. This study helped them to explore approaches that increase model robustness using structure-invariant word representations and robust training on noisy texts.<br>
<code>TODO: read paper</code></p>
<p>Rychalska et al performed in their <a href="https://pdfs.semanticscholar.org/8562/5148efbf0e556f9ae8145f0873e7c98a74bf.pdf">work</a> of ICNIP 2019 adversarial training on selected aspects and checked its transferability to the improvement of models with various corruption types.<br>
<code>TODO: read paper</code></p>
<h2 id="mldiag-1">MLDiag</h2>
<p>MLDiag introduces several model tests to ensure its robustness and disposal to go to production.<br>
We categorize theses tests into several categories and different scenario related to the model task.</p>
<h3 id="descriptive">Descriptive</h3>
<p>Descriptive tests highlight some model failures or success regarding dataset statistics.</p>

<table>
<thead>
<tr>
<th align="center">Test</th>
<th align="center">Target(s)</th>
<th align="center">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Length</td>
<td align="center">Text</td>
<td align="center">how data length impacts model errors (e.g. a text classification model always fails on short texts</td>
</tr>
<tr>
<td align="center">Similar</td>
<td align="center">text/Image</td>
<td align="center">does the model make always the same error (e.g. a classification model that fails in a whole category or in a cluster of very similar data</td>
</tr>
</tbody>
</table><h3 id="invariance">Invariance</h3>
<p>Inputs data are perturbed while measuring there impact on model prediction. This is closely related to data augmentation techniques, where we apply perturbations to inputs during training while preserving the original label. The problem with a <code>brut force</code> data augmentation is its computation cost. Using MLDiag tests, we can observe which perturbations affect the current model , we define then the set of augmentations that improve model consistency.</p>
<p>The list of model invariance tests is the following:</p>

<table>
<thead>
<tr>
<th align="center">Test</th>
<th align="center">Data</th>
<th align="center">Target(s)</th>
<th align="center">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Spell</td>
<td align="center">Text</td>
<td align="center">Classification</td>
<td align="center">how much spelling mistakes affect the model</td>
</tr>
<tr>
<td align="center">Pronoun</td>
<td align="center">Text</td>
<td align="center">Classification</td>
<td align="center">how much changing pronoun words impacts the model</td>
</tr>
<tr>
<td align="center">Sent-Order</td>
<td align="center">Text</td>
<td align="center">Classification</td>
<td align="center">how much sentence disorder  impacts the model</td>
</tr>
</tbody>
</table><h2 id="further-reading">Further reading</h2>
<p><strong>Blog posts</strong><br>
Jeremy Jordan, <a href="https://www.jeremyjordan.me/testing-ml/?utm_campaign=Data_Elixir&amp;utm_source=Data_Elixir_300">Effective testing for machine learning systems</a>, Aug 2020</p>
<p><strong>Papers</strong></p>
<ul>
<li><a href="https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf">Beyond Accuracy: Behavioral Testing of NLP Models with CheckList</a></li>
<li><a href="http://proceedings.mlr.press/v97/odena19a/odena19a.pdf">TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing</a></li>
<li><a href="http://idl.cs.washington.edu/files/2019-Errudite-ACL.pdf">Errudite: Scalable, Reproducible, and Testable Error Analysis</a></li>
<li><a href="https://arxiv.org/pdf/1806.00692.pdf">Stress Test Evaluation for Natural Language Inference</a></li>
<li><a href="https://arxiv.org/pdf/1711.02173.pdf">Synthetic and natural noise both break neural machine translation</a></li>
</ul>
<p><strong>Softwares</strong></p>
<ul>
<li><a href="https://github.com/uwdata/errudite">Errudites</a> an interactive tool for scalable, reproducible, and counterfactual error analysis of NLP models.</li>
<li><a href="https://github.com/MI2DataLab/WildNLP/tree/master/example">wildNLP</a> a python package to corrupt an input text and valuate NLP models’ robustness.</li>
<li><a href="https://github.com/marcotcr/checklist">CheckList</a> a tool that include multiple abstractions that help users generate large numbers of test cases easily, such as templates, lexicons, general-purpose perturbations, visualizations, and context-aware suggestions.</li>
</ul>

