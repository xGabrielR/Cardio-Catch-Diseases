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

- [x. References](#x-references)

---

<h2>0. Bussiness Problem</h2>
<hr>
<p><i>Cadio Catch Diseases is a company specializing in early-stage heart disease detection. Its business model is of the service type, that is, the company offers the early diagnosis of a cardiovascular disease for a certain price..</i></p>

<p>Currently, the diagnosis of a cardiovascular disease is done manually by a team of specialists. The current accuracy of the diagnosis varies between 55% and 65%, due to the complexity of the diagnosis and also to the fatigue of the team that takes turns to minimize the risks. The cost of each diagnosis, including the equipment and the analysts' payroll, is around R$ 1,000.00.</p>

> *With a Model, get a better precison on cardiovascular diagnosis.*
> 1. *What is the precision and accuracy of this new tool?*
> 2. *How mutch profit the Cardio Catch Diseases will earn with this new tool?* 
> 3. *What is the confidence interval of this new tool??*

<h3>0.1. What is a Service</h3>
<p>Service is a business model like consultory, the company make a work and receive a profit based on her work results. For example, Cardio Catch Diseases, the price of the diagnosis, paid by the client, varies according to the precision achieved by the time of specialists, the client pays R$500.00 for every 5% of accuracy above 50%. For example, for an accuracy of 55%, the diagnosis is R$500.00 for the client, for an accuracy of 60%, the value is R$1000.00, and so on. If the diagnostic accuracy is 50%, the customer does not pay for it.</p>
<p>Other example is terrain analysis, based on terrain size and terrain quality (terrain analysis) the price of the analysis can change severely.</p>

<h3>0.2. What is Cardiovascular Diseases</h3>
<p>Cardiovascular Diseases (CVD's) are a group of disorders of the heart and blood vessels.</p>
<p>Heart attacks and strokes are usually acute events and are mainly caused by a blockage that prevents blood from flowing to the heart or brain. The most common reason for this is a build-up of fatty deposits on the inner walls of the blood vessels that supply the heart or brain.</p>

<p>The most important behavioural risk factors of heart disease and stroke are unhealthy <strong>diet</strong>, <strong>physical inactivity</strong>, <strong>tobacco</strong> use and harmful use of <strong>alcohol</strong>. The effects of behavioural risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity. These “intermediate risks factors” can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.</p>

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
<p>Arrhythmia refers to an abnormal heart rhythm. There are various types of arrhythmias, The heart can beat too slow, too fast or irregularly, an arrhythmia can affect how well your heart works. With an irregular heartbeat, your heart may not be able to pump enough blood to meet your body’s needs</p>
<ul>
  <li>Bradycardia</li>
  <p>Heart rate that’s too slow, is when the heart rate is less than 60 beats per minute..</p>
  <li>Tachycardia</li>
  <p>Heart rate that’s too fast, refers to a heart rate of more than 100 beats per minute.</p>
</ul>

<h3>0.3. Blood Pressure</h3>


<p>The Dataset Base <a href='https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset'>Cardiovascular Disease</a>.</p>

<h2>1. Solution Strategy and Assumptions Resume</h2>
<hr>

<h3>1.1. First CRISP Cycle</h3>
<p>Under Desk Research Moment.</p>
<!-- <ul>
  <dl>
    <dt>Data Clearing & Descriptive Statistical.</dt>
      <dd>First real step is download the dataset, import in jupyter and start in seven steps to change data types, data dimension, fillout na... At first statistic dataframe, i used simple statistic descriptions to check how my data is organized, and check <strong>strong Unbalance Dataset.</strong></dd>
    <dt>Feature Engineering.</dt>
      <dd>In this step, with coggle.it to make a mind map and use the mind map to create some hypothesis list, after this list, i created some new features based on mdatetime.</dd>
    <dt>Data Filtering.</dt>
      <dd>TextArea</dd>
    <dt>Data Balance.</dt>
      <dd>TextArea</dd>
    <dt>Exploratory Data Analysis.</dt>
      <dd>TextArea</dd>
    <dt>Data Preparation.</dt>
      <dd>TextArea</dd>
    <dt>ML Models.</dt>
      <dd>TextArea</dd>
  </dl>
</ul> -->

<h2>x. References</h2>
<hr>


<ul>
  <li><a href='https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)'>Cardiovascular diseases</a></li>
  <li><a href='https://www.cdc.gov/heartdisease/about.htm'>Heart diseases</a></li>
  <li><a href='https://pubmed.ncbi.nlm.nih.gov/7285104/'>Negative Diastolic Pressure</a></li>
  <li><a href='https://www.mayoclinic.org/diseases-conditions/heart-valve-disease/symptoms-causes/syc-20353727'>Heart Valve</a></li>
  <li><a href='https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118'>Heart diseases</a></li>
  <li><a href='https://www.mayoclinic.org/diseases-conditions/heart-failure/symptoms-causes/syc-20373142'>Heart Failure</a></li>
</ul>
