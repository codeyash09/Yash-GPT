<i>Something new is on the horizon.</i>

OMEGA is close: the largest, fastest, and most ambitious model yet (converged to 4.02 loss in 2500 steps: >5x faster than beta). ~380 mil parameters
Release Date: Estimated to be Friday, June 26th.

<h1>Overview</h1>

Hi! Im Yash

I am working on building a block-based coding website where children can build transformers using blocks and I am trying to make it have 10 difficulty levels so it goes from surface level to true math: the rationale for this is that I want to teach kids at a young age how llms work because I learned coding from Scratch as a second grader and have self-studied to get to where I am, so imagine if a second grader learns backprop. The reason I made this is because of the quote attributed to Albert Einstein about if I can't explain it simply I don't understand it enough so I need to understand it as deeply as possible before I try to teach anyone.

I would have uploaded my "trained" weights too but github would not allow me to upload files that large.

<h2> Update: 3//26 </h2>
<img width="750" height="557" alt="image" src="https://github.com/user-attachments/assets/6974ec34-c680-485e-8c20-b26ab81b4f1a" />
<p>BPE allows for greater gains in less time. 6.11868 GLoss in 900 steps. </p>

<h2> Update: 3/1/26 </h2>
<p>Beta works however for faster runs, delete the input.txt file and rename "oldinput.txt" to "input.txt". Also now it makes loss curve pngs!!!</p>
<img width="631" height="478" alt="image" src="https://github.com/user-attachments/assets/d2784859-2204-4238-b697-e2d32d53f074" />
<p>Fresh start: 0.0001 lr</p>

<img width="588" height="435" alt="image" src="https://github.com/user-attachments/assets/28627947-59cd-4851-b5af-a7c5f004c1c2" />
<p>0.005 lr</p>

<img width="646" height="486" alt="image" src="https://github.com/user-attachments/assets/ac106634-71ea-46f3-b764-7ef2d14039b7" />
<p>PR: Longest run</p>


                                                               

The new code does not have that many comments on a lot of it because most of the code is from before and if you check the commented out segments, you will likely find explantations behind most of the code.


Do not be worried about even layers (0, 2, 4) having spikes (both above and below 40-60%) as the 0.92 decay on the stabilizing layers (1, 3, 5) will pull them back into the "Goldilocks" zone as the backprop and feed forward will cause the restrictions to flow into the even layers. Furthermore, the even layers are simply learning and exploring nueron usage and they will stabilize as model reaches equillibrium due to the nature of the alternating weight control system.

When I used a uniform decay, I encountered dying ReLU and without any decay, ReLU would gradually climb to above 70% and cause Loss to spike. Thus, I have found that this alternating method works best.

Best GLoss (As of 1/6/26 - ALPHA) 4.247961485781738
Current GLoss (2/9/26 - BETA) - 4.864489969095911

I built this over the course of winter break and did not make the repo until later on because I did not have a need to share this with anyone until then and also because I already locally backup my files.

Use alpha, beta is unstable and will slow down and completely stop improving at 5.95 (I am investigating this currently) ---> Guessed Culprit: Overfitting due to low data to parameters ratio.

Real culprit: clipping was not aligned and not consistent across forward and backprop

Anyways, I trained this on my rtx 4080 super with 64 gigs of ram and a ryzen 9 7950x3d on the pc I built over the last summer. Basically this is a warning because my GPU was at 99% usage with 7.6 gigs of its VRAM used up on one batch on beta (alpha is not as strenous) but I was also using 25.8 gigs of ddr5 ram. 
I'm saying this so you understand how much power it uses and I know that it does not work on all devices.


<h1>Quick Start</h1>

<p style="text-weight:bold;">Requires:</p>
<li>CuPy (NVIDIA GPU)</li>
<li>Python</li>
<li>NumPy</li>

<br>

<h3>FRESH START GUIDE ---> Run BPE.py and DO NOT CLOSE until Layer [X] ReLU Activity: [Y]% is printed. Then you may close it and start running beta.py</h3>
<h3>Training may take multiple days to train to a level of coherent generation </h3>
<p>Set train to True and let it run. Furthermore for alpha unlock dictionary by setting dictLock to False</p>
<p>Sidenote: I recommend only stopping training once it prints "[Checkpoint] Step ... | GLoss: ..." as it will not save otherwise</p>

<h2> Generation </h2>
<p>Turn off train (set train to False) at the start which will lead to generation</p>
<p>Sidenote: To change seed text, line 814 in alpha is calling the write function, simply change the text inside the "" to whatever seed text one prefers. Furthermore, one can adjust how many token will be generated using the second parameter of the write function.</p>
<p>Sidenote: beta's generation is rather slow but it has punctuation now but no post processing so it will not capitalize after a period, exclamation point, question mark, or any other end of sentence punctuation. Furthermore, punctuation is seperated by spaces in beta.</p>


<h1>STATS!!!</h1>
<h2>ALPHA</h2>
<li>512 dimensions</li>
<li>8 heads</li>
<li>6 layers</li>
<li>89,662,976 parameters</li>

<br>

<h2>BETA</h2> 
<li>1024 dimensions</li>
<li>16 heads</li>
<li>12 layers</li>
<li>292,584,448 parameters</li>
