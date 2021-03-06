# Cardio Catch Diseases

![cardio](https://user-images.githubusercontent.com/75986085/162591690-2bdb49a3-0136-4274-a1e1-e7cbbcd1dc31.png)

<h2>Summary</h2>
<hr>

- [0. Bussiness Problem](#0-bussiness-problem)
  - [0.1. What is a Service](#01-what-is-a-service)
  - [0.2. What is Cardiovascular Diseases](#02-what-is-cardiovascular-diseases)
    - [0.2.1. Heart Attack](#021-heart-attack) 
    - [0.2.2. Heart Failure](#022-heart-failure)
    - [0.2.3. Heart Valve Problems](#023-heart-valve-problems)
    - [0.2.4. Stroke](#024-stroke)
    - [0.2.5. Arrhythmia](#025-arrhythmia)

- [1. Solution Strategy & Assumptions Resume](#1-solution-strategy-and-assumptions-resume)
  - [1.1. First CRISP Cycle](#11-first-crisp-cycle)
  - [1.2. Second CRISP Cycle](#12-second-crisp-cycle)

- [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [2.1. EDA On First Cycle](#21-eda-on-first-cycle)
  - [2.2. Top 3 Eda Insights](#22-top-3-eda-insights)

- [3. Data Preparation](#3-data-preparation)
  - [3.1. Dataset Balance](#31-dataset-balance)
    - [3.1.1. First Cycle](#311-first-cycle)
    - [3.1.2. Second Cycle](#312-second-cycle) 
    - [3.1.3. Third Cycle](#313-third-cycle)  

- [4. Embedding Space Study](#4-embedding-space-study)

- [5. Machine Learning Models](#5-machine-learning-models)

- [6. Model Tuning](#6-model-tuning)
  - [6.1. First Cycle Model Tuning](#61-first-cycle-model-tuning)
    - [6.1.1. Calibration Curves](#611-calibration-curves)
    - [6.1.2. Confidence Intervals](#612-confidence-intervals) 
  - [6.2. Second Cycle Model Tuning](62-second-cycle-model-tuning)
    - [6.2.1. Calibration Curves](#621-calibration-curves)
    - [6.2.2. Confidence Intervals](#622-confidence-intervals) 

- [7. Model Bussiness Results](#6-model-bussiness-results)
  - [7.1. What is the precision and accuracy of this new tool](#71-what-is-the-precision-and-accuracy-of-this-new-tool) 
  - [7.2. How mutch profit the CCD will earn with this new tool](#72-how-mutch-profit-the-ccd-will-earn-with-this-new-tool) 
  - [7.3. What is the confidence interval of this new tool](#73-what-is-the-confidence-interval-of-this-new-tool)

- [8. Model Deployment](#8-model-deployment)

- [9. References](#9-references)

---

<h2>0. Bussiness Problem</h2>
<hr>
<p><i>Cadio Catch Diseases is a company specializing in early-stage heart disease detection. Its business model is of the service type, that is, the company offers the early diagnosis of a cardiovascular disease for a certain price.</i></p>

<p>Currently, the diagnosis of a cardiovascular disease is done manually by a team of specialists. The current accuracy of the diagnosis varies between 55% and 65%, due to the complexity of the diagnosis and also to the fatigue of the team that takes turns to minimize the risks. The cost of each diagnosis, including the equipment and the analysts' payroll, is around R$ 1,000.00.</p>

> *With a Model, get a better precison on cardiovascular diagnosis.*
> 1. *What is the precision and accuracy of this new tool?*
> 2. *How mutch profit the Cardio Catch Diseases will earn with this new tool?* 
> 3. *What is the confidence interval of this new tool?*

<h3>0.1. What is a Service</h3>
<p>Service is a business model like consultory, the company make a work and receive a profit based on her work results. For example, Cardio Catch Diseases, the price of the diagnosis, paid by the client, varies according to the precision achieved by the time of specialists, the client pays R$500.00 for every 5% of accuracy above 50%. For example, for an accuracy of 55%, the diagnosis is R$500.00 for the client, for an accuracy of 60%, the value is R$1000.00, and so on. If the diagnostic accuracy is 50%, the customer does not pay for it.</p>
<p>Other example is terrain analysis, based on terrain size and terrain quality (terrain analysis) the price of the analysis can change severely.</p>

<h3>0.2. What is Cardiovascular Diseases</h3>
<p>Cardiovascular Diseases (CVD's) are a group of disorders of the heart and blood vessels.</p>
<p>Heart attacks and strokes are usually acute events and are mainly caused by a blockage that prevents blood from flowing to the heart or brain. The most common reason for this is a build-up of fatty deposits on the inner walls of the blood vessels that supply the heart or brain.</p>

<p>The most important behavioural risk factors of heart disease and stroke are unhealthy <strong>diet</strong>, <strong>physical inactivity</strong>, <strong>tobacco</strong> use and harmful use of <strong>alcohol</strong>. The effects of behavioural risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity. These ???intermediate risks factors??? can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.</p>

<p>In addition, drug treatment of hypertension, diabetes and high blood lipids are necessary to reduce cardiovascular risk and prevent heart attacks and strokes among people with these conditions.</p>

<h4>0.2.1. Heart Attack</h4>
<p>A heart attack occurs when the blood flow to a part of the heart is blocked by a blood clot, fat or other substances. If this clot cuts off the blood flow completely, the part of the heart muscle supplied by that artery begins to die, if the blood flow is interrupted can damage or destroy part of heart muscle.</p>
<p>The medications and lifestyle changes that your doctor recommends may vary according to how badly your heart was damaged, and to what degree of heart disease caused the heart attack.</p>

<h4>0.2.2. Heart Failure</h4>
<p>Occurs when the heart muscle doesn't pump blood as well as it should. When this happens, blood often backs up and fluid can build up in the lungs, causing shortness of breath. In heart failure, the main pumping chambers of the heart (the ventricles) may become stiff and not fill properly between beats. In some people, the heart muscle may become damaged and weakened. The ventricles may stretch to the point that the heart can't pump enough blood through the body.</p>
<p>One way to prevent heart failure is to prevent and control conditions that can cause it, such as coronary artery disease, high blood pressure, diabetes and obesity.</p>

<h4>0.2.3. Heart Valve Problems</h4>
<p>Your heart has four valves that keep blood flowing in the correct direction. In some cases, one or more of the valves don't open or close properly. This can cause the blood flow through your heart to your body to be disrupted.</p>
<ul>
  <li>Regurgitation.</li>
  <p>The valve flaps don't close properly, causing blood to leak backward in your heart. This commonly occurs due to valve flaps bulging back, a condition called prolapse.</p>
  <li>Stenosis.</li>
  <p>The valve flaps become thick or stiff and possibly fuse together. This results in a narrowed valve opening and reduced blood flow through the valve.</p>
  <li>Atresia.</li>
  <p>The valve isn't formed, and a solid sheet of tissue blocks the blood flow between the heart chambers.</p>
</ul>

<h4>0.2.4. Stroke</h4>
<p>Have two types of Stroke:</p>
<ul>
  <li>Ischemic stroke</li>
  <p>It happens when the brain's blood vessels become narrowed or blocked, causing severely reduced blood flow (ischemia). Are caused by fatty deposits that build up in blood vessels or by blood clots or other debris that travel through the bloodstream, most often from the heart, and lodge in the blood vessels in the brain.</p>
  <li>Hemorrhagic stroke</li>
  <p>Occurs when a blood vessel within the brain bursts. This is most often caused by uncontrolled hypertension (high blood pressure).</p>
</ul>

<h4>0.2.5. Arrhythmia</h4>
<p>Arrhythmia refers to an abnormal heart rhythm. There are various types of arrhythmias, The heart can beat too slow, too fast or irregularly, an arrhythmia can affect how well your heart works. With an irregular heartbeat, your heart may not be able to pump enough blood to meet your body???s needs</p>
<ul>
  <li>Bradycardia</li>
  <p>Heart rate that???s too slow, is when the heart rate is less than 60 beats per minute.</p>
  <li>Tachycardia</li>
  <p>Heart rate that???s too fast, refers to a heart rate of more than 100 beats per minute.</p>
</ul>

<h3>0.3. Blood Pressure</h3>

![blood](https://user-images.githubusercontent.com/75986085/162626318-0e407579-cbc1-4378-9a8d-4a0df614a69b.png)

<p>The blood pressure is other important thing to check heart health, itsn very important to check the systolic and diastolic, like a fraction of blood pressure on mm Hg ( 120 systolic / 60 diastolic ).</p>

<ul>
  <li>Systolic Pressure</li>
  <p>The top number refers to the amount of pressure in your arteries during the contraction of your heart muscle.</p>
  <li>Diastolic Pressure</li>
  <p>The bottom number refers to your blood pressure when your heart muscle is between beats.</p>
</ul>

<p>The Dataset Base <a href='https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset'>Cardiovascular Disease</a>.</p>

<h2>1. Solution Strategy and Assumptions Resume</h2>
<hr>

<p>The Deployment of the model is on Google Sheets, the cardio team can check the probability of the new users on base have or no a 'Cardio Diseases'.</p>

![cardio_diseases_proba](https://user-images.githubusercontent.com/75986085/165198437-919ca29c-8ce5-4d55-b6de-0445dc451c89.gif)

<a href="https://github.com/xGabrielR/Cardio-Catch-Diseases/blob/main/Documenta%C3%A7%C3%A3o%20PT-BR.pdf">Full Documentation PT-BR</a>

<h3>1.1. First CRISP Cycle</h3>

<ul>
  <dl>
    <dt>Data Cleaning & Descriptive Statistical.</dt>
      <dd>First real step is download the dataset, import in jupyter and start in seven steps to change data types, data dimension, fillout na... At first statistic dataframe, i used simple statistic descriptions to check how my data is organized, and check, in dataset have only numerical attirbutes!</dd>
    <dt>Feature Engineering.</dt>
      <dd>In this step, with coggle.it to make a mind map and use the mind map to create some hypothesis list, after this list, i created some new features based on blood, like blood volume, blood systolic and diastolic pressure, pulse pressure and bmi, but on dataset do not have other features for more feature engineering.</dd>
    <dt>Data Filtering.</dt>
      <dd>On Dataset Have some Outliers, height, weight, blood pressure, extreme negative diastolic pressure, etc,to work with this i have tried to get a "medical intuition" and removed extreme negative diastolic and systolic blood pressure, and a little height and weight threshold.</dd>
    <dt>Data Balance.</dt>
      <dd>On Next Cycle i like to use SMOTEEN to clean data overlapping for better model accuracy and precision.</dd>
    <dt>Exploratory Data Analysis.</dt>
      <dd>With this dataset is hard to define a class limit, need much deeper feature engineering.</dd>
    <dt>Data Preparation.</dt>
      <dd>Used MinMiaxScaler, Robust Scaler & Frequency Encoding for Rescaling some features and drop "alco" & "smoke", because XGBoost and RF did not classify these two features as relevant.</dd>
    <dt>ML Models.</dt>
      <dd>I try 7 models on total, four are Tree-based models.</dd>
  </dl>
</ul>

<h3>1.2. Second CRISP Cycle</h3>

<ul>
  <dl>
    <dt>Data Balance.</dt>
      <dd>I used SMOTEEN and SMOTETOMEK for Dataset Balance.</dd>
    <dt>Data Preparation.</dt>
      <dd>Used MinMiaxScaler, Robust Scaler & Frequency Encoding for Rescaling both datasets (Smoteen Dataset and Smotetomek Dataset) some features and drop "alco" & "smoke", because XGBoost and RF did not classify these two features as relevant.</dd>
    <dt>ML Models.</dt>
      <dd>I Used SGD and Ada, focus on SGD classifier.</dd>
  </dl>
</ul>

<h3>3.1.3. Third Cycle</h3>

<ul>
  <dl>
    <dt>Feature Space Study.</dt>
      <dd>In this new cycle, i have try some Embedding spaces using UMAP, PCA, tSNE and Tree-Based Embedding.</dt>
      <dd>In this step, I used all the datasets that I had created previously, like the entire dataset, Smoteen and Smotetomeklinks dataset, to analyze all these data behaviors with different tools.</dd>
  </dl>
</ul>


<h2>2. Exploratory Data Analysis</h2>
<hr>
<p>EDA is the most important step on Data Science projects, in this step you "deep dive" on data and work with univariable, bivariable and multivariable data analysis.</p>

<h3>2.1. EDA On First Cycle</h3>

<p>In Univariable Analisys</p>
<ol>
    <li>The Dataset have some identical and normal features, good for machine learning model.</li>
    <li>Have a good balance between classes.</li>
</ol>

<p>Bivariate Analysis</p>
<ol>
  <li>The Features based on class, it's hard to find a separating boundary.</li>
</ol>

<p>With Pearson's correlation method, i get aprox .50 positive correlation with height and gender!</p>


<h3>2.2. Top 3 Eda Insights</h3>
<hr>

<p>People who suffer from dwarfism have 25% higher cholesterol than a normal adult person.</p>

![nanism](https://user-images.githubusercontent.com/75986085/162857654-59101da3-ccbd-4d93-8c26-8bcfb2811509.png)


<p>Alcoholic people have a greater chance of developing cardiovascular disease than people who smoke.</p>

![alc](https://user-images.githubusercontent.com/75986085/162857750-8f0f7dbc-fea0-4bec-90b3-e1f14210e1d9.png)


<p>People over 45 are 70% more likely to develop cardiovascular disease.</p>

![cardiovascular](https://user-images.githubusercontent.com/75986085/162857536-57ea4ac7-0b9c-4e60-9f52-e58294bae3c6.png)


<h2>3. Data Preparation</h2>
<hr>

<p>For Rescaling i used both, MinMax and RobustScaler and Frequency Encoding for numerical features like Gluc Level.</p>
<p>On first cycle i did not used Smoteen for cleaning data overlapping, in next cycles i will go try more things like better feature engineering, PCA, Smoteen...</p>

<h3>3.1. Dataset Balance</h3>

<h4>3.1.1. First Cycle</h4>
<p>On Next Cycle i will try balance Dataset with Smoteen and Smotetomeklinks.</p>

<h4>3.1.2. Second Cycle</h4>
<p>I Try both, Smoteen and Smotetomek on Cardio Dataset.</p>
<p>Smoteen removed a lot of data overlapping on Dataset and Smotetomek do not work much well than Smoteen for Balance.</p>

![dataset_balance](https://user-images.githubusercontent.com/75986085/164985257-bb1178e7-6fd5-497c-a930-93fccd98ebca.png)


<h2>4. Embedding Space Study</h2>
<hr>

<p>The Feature Space study or Embedding study, in this step i have dedicated a complete Cycle, the Third Cycle, to analyze all data behaviors with different datasets that i have already created the Smoteen Dataset, Smotetomeklinks Dataset and Full Dataset with diferent tools and methods.</p>
<p>For Rescaling i used Both, StandardScaler and MinMaxScaler to analyze this differences.</p>
<p>With this both Rescaling methods i performed the Umap, tSNE, PCA and Tree-Based Embedding on all Three Datasets, you can check the Notebook of Third Cycle to see the different behaviors.</p>
<p>I'm not going to share all the spaces in the README so as not to make it too big.</p>

![pca_tree_based_embedding_smoteen_dataset](https://user-images.githubusercontent.com/75986085/165412167-2327e719-9d3e-4276-a9e4-c0541bae1e1c.gif)

<h2>5. Machine Learning Models</h2>
<hr>

![models](https://user-images.githubusercontent.com/75986085/163059972-193109c1-bca2-4b89-ad43-9276d6848b5b.png)

<ul>
   <li>Support Vector Machines</li>
   <p>I studied about the power of SVM, but in training I didn't see it that powerful, maybe I'll proceed with this model to tune.</p>
   <li>XGBoost</li>
   <p>My personal favorite model, fast, light and haved a normal results on training with this dataset.</p>
   <li>Random Forest</li>
   <p>Random Forest get less results than XGBoost, but, rf have selected some important features thai i selected.</p>
   <li>K Nearest Neighbors</li>
   <p>First time I trained a KNN, I really liked the result with 10 neighbors.</p>
   <li>Stochastic Gradient Descent</li>
   <p>This is a good "linear" model, maybe i use on tuning too.</p>
   <li>Light GBM</li>
   <p>Similar to XGBoost, but significantly better with this only one train dataset.</p>
   <li>Ada Boost</li>
   <p>First time trained AdaBoost.</p>
</ul>

<h2>6. Model Tuning</h2>
<hr>

<p>This is the principal step on this Data Science Project, because, there aren't many ways to create features or collect new data, so a very very *very* detailed data preparation and a good tuning in these cases is very important.</p>

<h3>6.1. First Cycle Model Tuning</h3>
<p>For First Cycle i used SGD and Ada Boosting for Tuning, because Ada have great performace and SGD is a ""linear model"" with linear coeficients. But after tuning, i chosed the SGD because he is it is much lighter on HD than ada, '5Kb' of Disk Space.</p>

![tuned](https://user-images.githubusercontent.com/75986085/164907644-5b0c314a-113a-4eaf-adce-66d6383ec3b7.png)

<p>I used a simple Random Search to find the best params for model.</p>

![cross](https://user-images.githubusercontent.com/75986085/164908814-b89026f2-dcd5-41c9-b243-659ae9b357f8.png)

<p>On Cross Validation the model have a good performace (Precision).</p>

<h4>6.1.1. Calibration Curves</h4>
<p>This step is after tuning the model, to calibrate the super and sub estimation adjustments.</p>

![cali](https://user-images.githubusercontent.com/75986085/164909008-ffd120d9-fd0f-4477-97f0-8c5f1384c20f.png)

<p>The Calibrated SGD simple performace.</p>

![final_model](https://user-images.githubusercontent.com/75986085/164909033-6786535f-4306-4695-852a-b6c888986f00.png)

<h4>6.1.2. Confidence Intervals</h4>
<p>This is the last step of the step of tuning the machine learning model, in this step the confidence intervals are calculated using a ready-made formula from MachineLearningMastery</p>

![boot](https://user-images.githubusercontent.com/75986085/164909015-8117ff1f-b909-4ccc-9478-2aea16b514aa.png)

<h3>6.2. Second Cycle Model Tuning</h3>
<p>I Using SGD and Ada on Second Cycle Too for tuning on Smoteen and Smotetomek Dataset. But after some tests i prefer to use ADA to production.</p>

<p>Final ADA Model Performace on Cycle II for Production.</p>

![ada_new](https://user-images.githubusercontent.com/75986085/165006808-9929673a-2dda-445e-b7b5-3a16cb8e5c18.png)

<h4>6.2.1. Calibration Curves</h4>
<p>The calibration curve of Raw ADA model</p>

![calibrated_ada_bootstrap](https://user-images.githubusercontent.com/75986085/165102550-0988868d-758b-43be-9ca6-b357b837f2be.png)

<p>The calibration curve of Tuned ADA model</p>

![calibrated_ada](https://user-images.githubusercontent.com/75986085/165006855-aa0d7425-c39b-488a-8ecb-0c0ab54da15a.png)

<h4>6.2.2. Confidence Intervals</h4>

<p>The Bootstrap of Tuned Only ADA Model.</p>

![raw_ada](https://user-images.githubusercontent.com/75986085/165006883-f71bcd9e-fad3-4cf8-afd4-702b57cdd4ce.png)

<p>I do not selected calibrated + tuned model because on bootstrap eith calibrated + tuned model i get an insignificantly larger error. I only used Tuned Model to Deploy.</p>

<h2>7. Model Bussiness Results</h2>
<p>Need to answer the Questions</p>

> 1. *What is the precision and accuracy of this new tool?*
> 2. *How mutch profit the Cardio Catch Diseases will earn with this new tool?* 
> 3. *What is the confidence interval of this new tool?*

<h3>7.1. What is the precision and accuracy of this new tool</h3>

At Cross Validation Between ( Mean + / - Std )

1. Accuracy ( 0.724 + / - 0.0006 )
2. Precision ( 0.7874 + / - 0.0048)

<h3>6.2. How mutch profit the CCD will earn with this new tool</h3>
<p>Based on All Dataset (68k Patients).</p>

<table>
  <tr>
    <td>% of Precision</td>
    <td>50 %</td>
    <td>55 %</td>
    <td>60 %</td>
    <td>65 %</td>
    <td>70 %</td>
    <td>75 %</td>
    <td>80 %</td>
  </tr>
  <tr>
    <td>Money / Precision</td>
    <td>FREE</td>
    <td>$ 500</td>
    <td>$ 1000</td>
    <td>$ 1500</td>
    <td>$ 2000</td>
    <td>$ 2500</td>
    <td>$ 3000</td>
  </tr>
  <tr>
    <td>Actual Money / Precision</td>
    <td>$ 0</td>
    <td>$ 343,530.00</td>
    <td>$ 687,060.00</td>
    <td>$ 1,030,590.00</td>
    <td>$ --#--</td>
    <td>$ --#--</td>
    <td>$ --#--</td>
  </tr>
    <td>Model Money / Precision</td>
    <td>$ --#--</td>
    <td>$ 343,530.00</td>
    <td>$ 687,060.00</td>
    <td>$ 1,030,590.00</td>
    <td>$ 1,374,120.00</td>
    <td>$ 1,717,650.00</td>
    <td>$ --#-- </td>
  </tr>
</table>

<table>
  <tr>
    <th>__ / __</th>
    <th>Best Scenario</th>
    <th>Worst Scenario</th>
  </tr>
  <tr>
    <td>Model</td>
    <td>+/- $ 1,717,650.00</td>
    <td>+/- $ 1,374,120.00</td>
  </tr>
  <tr>
    <td>Actual</td>
    <td>+/- $ 1,030,590.00</td>
    <td>+/- $ 343,530.00</td>
  </tr>
</table>

<h3>7.3. What is the confidence interval of this new tool</h3>

- **25%** confidence interval of Model Performace **( 77.81%  &  78.34% )**
- **50%** confidence interval of Model Performace **( 77.44%  &  78.59% )**
- **75%** confidence interval of Model Performace **( 77.02%  &  78.95% )**

<h2>8. Model Deployment</h2>
<hr>

<p>The Model is on Google Sheets, because it's a good deploy strategy and it also has a good benefit as for the study of cardiovascular disease cases and checking if the model needs improvement, example, the cardio team like to change the values to study new behaviors and how the model will classify against this new behavior</p>
<p>This is the step for user use the Model Prediction to make better decisions, i chose to deploy on Heroku and make a JS Request on Google Scripts.</p>

https://user-images.githubusercontent.com/75986085/165198901-af6ed758-a232-4670-ba62-e02819e035f8.mp4

<h2>9. References</h2>
<hr>

<ul>
  <li><a href='https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)'>Cardiovascular diseases</a></li>
  <li><a href='https://www.cdc.gov/heartdisease/about.htm'>Heart diseases</a></li>
  <li><a href='https://pubmed.ncbi.nlm.nih.gov/7285104/'>Negative Diastolic Pressure</a></li>
  <li><a href='https://www.mayoclinic.org/diseases-conditions/heart-valve-disease/symptoms-causes/syc-20353727'>Heart Valve</a></li>
  <li><a href='https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118'>Heart diseases</a></li>
  <li><a href='https://www.mayoclinic.org/diseases-conditions/heart-failure/symptoms-causes/syc-20373142'>Heart Failure</a></li>
  <li><a href='https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/'>Calibration Curves (MLM)</li>
</ul>
