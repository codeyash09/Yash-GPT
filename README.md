Hi! Im Yash

I wanted to make an slm and trained it on kid's stories...

I would have uploaded my "trained" weights too but github would not allow me to upload files that large.


The new code does not have that many comments on a lot of it because most of the code is from before and if you check the commented out segments, you will likely find explantations behind most of the code.


Do not be worried about even layers (0, 2, 4) having spikes (both above and below 40-60%) as the 0.92 decay on the stabilizing layers (1, 3, 5) will pull them back into the "Goldilocks" zone as the backprop and feed forward will cause the restrictions to flow into the even layers. Furthermore, the even layers are simply learning and exploring nueron usage and they will stabilize as model reaches equillibrium due to the nature of the alternating weight control system.

When I used a uniform decay, I encountered dying ReLU and without any decay, ReLU would gradually climb to above 70% and cause Loss to spike. Thus, I have found that this alternating method works best.

Current GLoss (As of 1/6/26) 4.247961485781738


I built this over the course of winter break and did not make the repo until later on because I did not have a need to share this with anyone until then and also because I already locally backup my files.
