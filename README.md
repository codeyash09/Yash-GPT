Hi! Im Yash

I wanted to make an slm and trained it on kid's stories...

I would have uploaded my "trained" weights too but github would not allow me to upload files that large.


The new code does not have that many comments on a lot of it because most of the code is from before and if you check the commented out segments, you will likely find explantations behind most of the code.


Do not be worried about even layers (0, 2, 4) having spikes (both above and below 40-60%) as the 0.92 decay on the stabilizing layers (1, 3, 5) will pull them back into the "Goldilocks" zone as the backprop and feed forward will cause the restrictions to flow into the even layers. Furthermore, the even layers are simply learning and exploring nueron usage and they will stabilize as model reaches equillibrium due to the nature of the alternating weight control system.

When I used a uniform decay, I encountered dying ReLU and without any decay, ReLU would gradually climb to above 70% and cause Loss to spike. Thus, I have found that this alternating method works best.

Lastest output from training run on 1/5/26:
Progress: |                    | 0.0% | 4.248489879339706 | True | Layer 5 ReLU Activity: 50.20%
Progress: |                    | 0.0% | 4.257713806360974 | True | Layer 4 ReLU Activity: 40.37%
Progress: |                    | 0.0% | 4.259700115654552 | True | Layer 3 ReLU Activity: 49.05%
Progress: |                    | 0.0% | 4.258158438159225 | True | Layer 2 ReLU Activity: 51.41%
Progress: |                    | 0.0% | 4.256423889232142 | True | Layer 1 ReLU Activity: 49.25%
Progress: |                    | 0.1% | 4.256166509833743 | Truee
[Checkpoint] Step 10000 | GLoss: 4.2562
 | Layer 0 ReLU Activity: 50.22%
Progress: |                    | 0.1% | 4.256412587463694 | True | Layer 5 ReLU Activity: 49.21%
Progress: |                    | 0.1% | 4.2576678271805255 | True | Layer 4 ReLU Activity: 41.41%
Progress: |                    | 0.1% | 4.257737496273611 | True | Layer 3 ReLU Activity: 50.91%
Progress: |                    | 0.1% | 4.258324036893603 | True | Layer 2 ReLU Activity: 53.78%
Progress: |                    | 0.1% | 4.258223072646965 | True | Layer 1 ReLU Activity: 50.03%
Progress: |                    | 0.1% | 4.2568525686579015 | True
[Checkpoint] Step 20000 | GLoss: 4.2569
 | Layer 0 ReLU Activity: 55.19%
Progress: |                    | 0.1% | 4.255925357694626 | True | Layer 5 ReLU Activity: 50.05%
Progress: |                    | 0.1% | 4.254658893599823 | True | Layer 4 ReLU Activity: 42.66%
Progress: |                    | 0.1% | 4.254187517820671 | True | Layer 3 ReLU Activity: 51.09%
Progress: |                    | 0.1% | 4.254046714973297 | True | Layer 2 ReLU Activity: 51.61%
Progress: |                    | 0.1% | 4.2535500212035755 | True | Layer 1 ReLU Activity: 49.59%
Progress: |                    | 0.2% | 4.253238871879524 | Truee
[Checkpoint] Step 30000 | GLoss: 4.2532
 | Layer 0 ReLU Activity: 52.15%
Progress: |                    | 0.2% | 4.252630961199052 | True | Layer 5 ReLU Activity: 49.95%
Progress: |                    | 0.2% | 4.2521792004003744 | True | Layer 4 ReLU Activity: 40.96%
Progress: |                    | 0.2% | 4.251517444128161 | True | Layer 3 ReLU Activity: 50.67%
Progress: |                    | 0.2% | 4.251034806741396 | True | Layer 2 ReLU Activity: 51.79%
Progress: |                    | 0.2% | 4.251037274583022 | True | Layer 1 ReLU Activity: 50.47%
Progress: |                    | 0.2% | 4.2509371965082545 | True
[Checkpoint] Step 40000 | GLoss: 4.2509
 | Layer 0 ReLU Activity: 54.05%
Progress: |                    | 0.2% | 4.250378139974014 | True | Layer 5 ReLU Activity: 49.90%
Progress: |                    | 0.2% | 4.250009881335297 | True | Layer 4 ReLU Activity: 41.87%
Progress: |                    | 0.2% | 4.250026057851739 | True | Layer 3 ReLU Activity: 49.63%
Progress: |                    | 0.2% | 4.249477955013982 | True | Layer 2 ReLU Activity: 52.02%
Progress: |                    | 0.2% | 4.248632234427803 | True | Layer 1 ReLU Activity: 50.35%
Progress: |                    | 0.3% | 4.247969863815738 | Truee
[Checkpoint] Step 50000 | GLoss: 4.2480
 | Layer 0 ReLU Activity: 56.60%
Progress: |                    | 0.3% | 4.247961485781738 | True
