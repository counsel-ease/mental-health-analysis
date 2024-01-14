High and Low Quality Counseling Dataset
===================================================================
Verónica Pérez-Rosas, Xinyi Wu, Kenneth Resnicow, Rada Mihalcea
University of Michigan

vrncapr@umich.edu
wuxinyi@umich.edu
kresnic@umich.edu
mihalcea@umich.edu

Version 1.0
July 2019
-----------------------------------------------------------------
1. Introduction
This document describes the dataset used in the paper: What Makes a Good Counselor? Learning to Distinguish between High-quality and Low-quality Counseling Conversations (Perez-Rosas et. al, 2019).

-----------------------------------------------------------------
2. Content

The archive contains three folders and one README file. The README (this file) contains information about the data collected and their sources as well as citation and acknowledgements information.
The folders are as follows:
- Transcriptions: The folder includes the transcripts of counseling conversations in the dataset. 
- labels.csv: a csv file containing the id of the transcript and its corresponding label.
- urls.csv: a csv file pairing the transcript id and the video url in YouTube  (whenever available). Urls retrieved  as of July 15th, 2019.
-----------------------------------------------------------------
3. Dataset Information
The dataset consists of counseling videos that are publicly available on YouTube channels and other public websites.  The set of videos consist of Motivational Interviewing (MI) counseling demonstrations by professional counselors and MI role-play counseling by psychology students. Each video portrays different speakers and the conversations cover various health topics including smoking cessation, alcohol consumption, substance abuse, weight management, and medication adherence. More details on the data collection process are available in the paper that introduced the dataset (Perez-Rosas et al, 2019; full reference below).

The dataset consists of 259 counseling conversations, with 155 video clips labeled as high-quality counseling and 104 labeled as low-quality counseling. The length of the conversations in the dataset ranges from 5-20 minutes.

-----------------------------------------------------------------
5. Transcripts 

Transcriptions of the videoclips were collected using YouTube automatic captioning and Mechanical Turk whenever automatic transcription was not available. Speakers in the conversations are labeled manually and also following a heuristic to label the speaking sequence. 

-----------------------------------------------------------------
6. Feedback
For further questions or inquiries about this dataset, you can contact: Veronica Perez-Rosas or Rada Mihalcea
vrncapr@umich.edu
mihalcea@umich.edu

-----------------------------------------------------------------
7. Citation Information
Bibtex:
@article{Rosas19What,
author = {P\'{e}rez-Rosas, Ver\’{o}nica, Xinyi Wu, Kenneth Resnicow, and Rada Mihalcea},
title = {What Makes a Good Counselor? Learning to Distinguish between High-quality and Low-quality Counseling Conversations},
journal = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
year = {2019}
}


Text:
Verónica Pérez-Rosas, Xinyi Wu, Kenneth Resnicow, Rada Mihalcea. 2019. What Makes a Good Counselor? Learning to Distinguish between High-quality and Low-quality Counseling Conversations. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL). Florence, Italy. 

-----------------------------------------------------------------
8. Acknowledgements
This material is based in part upon work supported by the Michigan Institute for Data Science, by the National Science Foundation (grant #1815291), and by the John Templeton Foundation (grant #61156). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author and do not necessarily reflect the views of the Michigan Institute for Data Science, the National Science Foundation, or John Templeton Foundation. 
