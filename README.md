<h1>Block-Based Transformer Educational Tool</h1>

<blockquote>"If you can't explain it simply, you don't understand it well enough." — Albert Einstein</blockquote>

<p>Welcome. I am Yash. I built this project to teach children how Large Language Models (LLMs) work from the ground up. Inspired by my own journey of learning to code on Scratch in the second grade, I am developing a block-based coding environment with 10 difficulty levels. This curriculum is designed to take students from surface-level concepts all the way to understanding true mathematical backpropagation.</p>

<p>To teach this effectively, I built and trained this transformer architecture from scratch to ensure a deep, foundational understanding of the underlying mechanics.</p>

<hr>

<h2>Coming Soon: Project OMEGA</h2>
<p><b>Release Date:</b> Estimated Friday, June 26th</p>
<p><i>Something new is on the horizon.</i> OMEGA is our largest, fastest, and most ambitious model yet.</p>
<ul>
<li><b>Parameters:</b> ~380 Million</li>
<li><b>Performance:</b> Converged to 4.02 loss in 2500 steps (>5x faster than Beta).</li>
</ul>

<hr>

<h2>Model Architecture and Statistics</h2>
<p><i>Note: Pre-trained weights are not hosted on GitHub due to file size limits.</i></p>

<h3>ALPHA (Stable / Recommended)</h3>
<ul>
<li><b>Dimensions:</b> 512</li>
<li><b>Attention Heads:</b> 8</li>
<li><b>Layers:</b> 6</li>
<li><b>Total Parameters:</b> 89,662,976</li>
<li><b>Best GLoss:</b> 4.247961485781738</li>
</ul>

<h3>BETA (Development)</h3>
<ul>
<li><b>Dimensions:</b> 1024</li>
<li><b>Attention Heads:</b> 16</li>
<li><b>Layers:</b> 12</li>
<li><b>Total Parameters:</b> 292,584,448</li>
<li><b>Current GLoss:</b> 4.864489969095911</li>
</ul>

<p><b>A Note on Beta's Stability:</b> I am currently investigating a bug where Beta stops improving at a loss of 5.95. The root cause was identified as clipping that was inconsistent across forward and backpropagation, leading to overfitting due to a low data-to-parameters ratio. Alpha remains the recommended and most stable build.</p>

<hr>

<h2>System Requirements and Hardware Warning</h2>
<p>Training these models requires significant computing power. Please review the hardware parameters before proceeding.</p>

<p><b>Software Requirements:</b></p>
<ul>
<li>Python</li>
<li>NumPy</li>
<li>CuPy (Requires NVIDIA GPU)</li>
</ul>

<p><b>Hardware Profile (Development Machine):</b></p>
<ul>
<li><b>GPU:</b> RTX 4080 Super (16GB VRAM)</li>
<li><b>CPU:</b> Ryzen 9 7950X3D</li>
<li><b>RAM:</b> 64GB DDR5</li>
</ul>

<p><b>Warning:</b> A single batch on Beta utilized 99% of GPU compute, 7.6GB of VRAM, and 25.8GB of system RAM. Ensure your system can handle these loads before initiating training.</p>

<hr>

<h2>Quick Start Guide</h2>

<h3>1. Initial Setup and Byte-Pair Encoding (BPE)</h3>
<p><b>FRESH START GUIDE:</b> Before running the main model, you must initialize the BPE.</p>
<ul>
<li>Run <code>BPE.py</code>.</li>
<li><b>Do not close</b> the script until the console prints: <code>Layer [X] ReLU Activity: [Y]%</code>.</li>
<li>Once printed, you may safely close the script and run <code>beta.py</code> (or <code>alpha.py</code>).</li>
</ul>
<p><i>Tip for Faster Runs (Beta):</i> Delete the <code>input.txt</code> file and rename <code>oldinput.txt</code> to <code>input.txt</code>.</p>

<h3>2. Training</h3>
<ul>
<li>Set <code>train = True</code> in the script to begin.</li>
<li>For Alpha: Unlock the dictionary by setting <code>dictLock = False</code>.</li>
<li><b>Important:</b> Training may take multiple days to achieve coherent text generation. Only interrupt training <i>after</i> you see <code>[Checkpoint] Step ... | GLoss: ...</code> in the console, otherwise your progress will not save.</li>
</ul>

<h3>3. Generation</h3>
<ul>
<li>Set <code>train = False</code> in the script at the start.</li>
<li><b>Seed Text:</b> To change the seed text in Alpha, navigate to line 814 and modify the text string inside the <code>write()</code> function. The second parameter controls the token generation count.</li>
<li><b>Note on Beta Generation:</b> Beta's generation is currently slower but supports punctuation (separated by spaces). It lacks post-processing, meaning it will not auto-capitalize after end-of-sentence punctuation.</li>
</ul>

<hr>

<h2>Technical Notes: Alternating Weight Control</h2>
<p>When reviewing the output, you will notice spikes in even layers (0, 2, 4)—both above and below 40-60%. <b>This is expected behavior.</b></p>
<p>Initially, implementing a uniform decay resulted in dying ReLUs. Conversely, applying no decay caused ReLUs to climb above 70%, resulting in massive Loss spikes. To solve this, I implemented an alternating weight control system. The even layers are allowed to explore neuron usage, while a 0.92 decay on the stabilizing odd layers (1, 3, 5) pulls them back into the "Goldilocks" zone during feed-forward and backpropagation. The model will naturally stabilize as it reaches equilibrium.</p>

<hr>

<h2>Development Updates and Loss Curves</h2>

<h3>Update: 3/26/26</h3>
<p>BPE allows for greater gains in less time. Achieved 6.11868 GLoss in 900 steps.</p>
<img width="750" height="557" alt="image" src="https://github.com/user-attachments/assets/6974ec34-c680-485e-8c20-b26ab81b4f1a" />

<h3>Update: 3/1/26</h3>
<p>Beta is operational and now generates loss curve PNGs automatically.</p>

<p><b>Fresh start: 0.0001 lr</b></p>
<img width="631" height="478" alt="image" src="https://github.com/user-attachments/assets/d2784859-2204-4238-b697-e2d32d53f074" />

<p><b>0.005 lr</b></p>
<img width="588" height="435" alt="image" src="https://github.com/user-attachments/assets/28627947-59cd-4851-b5af-a7c5f004c1c2" />

<p><b>Personal Record: Longest run</b></p>
<img width="646" height="486" alt="image" src="https://github.com/user-attachments/assets/ac106634-71ea-46f3-b764-7ef2d14039b7" />
