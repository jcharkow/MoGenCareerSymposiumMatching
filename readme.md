# **MoGen Career Day Symposium Matching Algorithm**


## **Conference Setup**
Students are matched with mentors (1-many) for three roundtable sessions. Ideally each table has at least 2 students and no more than 10 students. 


## **Data Type**
Students rank as many/as few mentors as they want as either strongest interest (1), strong interest (2) or interested (3). Mentors have no preference.


## **Current Limitations**
Students are only matched with mentors they have requested meaning that the algorithm cannot handle students with less than 3 prefrences. Furthermore, students with less preferences are more likely to be matched with their top preferences as the algorithm cannot match them otherwise. 

## **Algorithm Parameters**
### **Mentor Popularity Parameters**
 To determine mentor popularity, mentors as awarded a certain number of points based on how many and how high students rank them
- RANK_1_PTS = points awarded to mentor if a student ranks them as "strongest interest"
- RANK_2_PTS = points awarded to mentor if a student ranks them as "strong interest"
- RANK_3_PTS = points awarded to mentor if a student ranks them as "interested"


### **Matching Parameters**
- MAX_PER_GROUP = Maximum number of students in initial round of pairing
- HARD_MAX_PER_GROUP = maximum number of students in secondary paring round (only performed if initial matching fails)




## **Algorithm Implementation**
This is a greedy algorithm meaning that once students are matched to a mentor group, this match is not changed. 
#### **Part A: Determine Mentor Popularity**
1. Score each mentor, mentors are awarded points based on how many and how high students rank them.
	2. For each Student that has Strongest Interest award RANK_1_PTS (default 10)
	3. For each Student that has Strong Interest award RANK_2_PTS (default 5)
	3. For each Student that has Interested award RANK_3_PTS (default 1)
#### **Part B: Pair each student with a mentorship group**
1. Sort students based on their priority students graduating soon have priority over lower year students
2. For Each Student: (From highest to lowest priority)
	2. Sort preferences by rank (highest to lowest) and mentor popularity (lowest to highest)
		3. For Each preference try pairing. Pairing is sucessful if there are less than MAX_PER_GROUP present. Keep going down the list of mentors until a pair is found
		3. If pairing is still unsucessful rety with the HARD_MAX_PER_GROUP cutoff.	

### **Part C: Update Preferences for next round**
1. Remove the preference that the student is paired with to ensure that a student is not paired with the same mentor twice. 


The overall workflow for 3 roundatable sessions is: A-B1-C1-B2-C2-B3
