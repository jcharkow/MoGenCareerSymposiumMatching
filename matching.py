### This stores all the functions for matching
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


class Preferences():
    ''' 
    This is the student preferences object.
    We can do manipulations on the students preferences table
    
    Class Members
    df = index is the email and columns are mentors, each value is a preference (represented as a number)
    metaData = metaData where index is an email
    
    If metadata is to be determined automatically, the provided df must have the columns: Group, Year, Timestamp in order to determine priority
    
    If df is presupplied should be sorted in order of priority
    '''
    
    
    def __init__(self, df, metaData=None):
        '''
        Initialize the object 
        '''
        
        if metaData is not None:
            print("Metadata set, assuming dataframe is already formatted") 
            self.df = df
            self.metaData = metaData
        
        else:
            self.metaData, self.df = Preferences.format_df(df)
    
    @staticmethod
    def getGroup(df):
        ''' 
        This function maps a group to a prioirty, we want to prioritize upper year students
        order is
        1: Grads 6, 6+
        2: Grads 5, Grad 2 with MSc.
        3: Grads 3, 4, postdocs
        4: Other Mogen Grad Students
        5: Non Mogen students, Undergrads
        
        input: dataframe with columns 'Group' and 'Year' 
        '''
        if df['Year'] == 'Year 2, planning to graduate with an MSc':
            year = 2
            group = df['Group'] + " (MSc)"
        
        elif isinstance(df['Year'], float) and df['Year'] is np.nan:
            year = 0
            group = df['Group']
        else:
            year = int(re.findall(r'\d', df['Year'])[0])
            group = df['Group']
            
        
        if group == 'MoGen research graduate student' and year >= 6:
            return 1
        
        elif group == "MoGen research graduate student" and year == 5: 
            return 2
        elif group=="MoGen research graduate student (MSc)": # 2nd year planning on graduating with Masters
            return 2
        
        elif group=="MoGen MHSc student":
            return 2
        
        elif group=='Medical Genomics graduate student':
            return 2
        
        
        elif group == "MoGen research graduate student" and year >= 3 and year <= 4: 
            return 3
        elif group == "Postdoc":
            return 3
        elif group == "MoGen research graduate student":
            return 4
        elif group == "Mogen MHSc alumna":
            return 4
        elif group == "Trainee at a department other than MoGen":
            return 5
        elif group == 'Trainee at a department other than MoGen (MSc)':
            return 5
        elif group == "MoGen undergraduate student":
            return 6
        elif group == "Biochem undergraduate student in a MoGen lab (I wasn't sure which category this was).":
            return 6
        elif group == "Graduating undergraduate student from a non-MoGen program (HMB & CSB)":
            return 6

        else:
            return 6
            raise Exception("group: {}, year: {} not matched".format(group, year))
    
    
    @staticmethod
    def format_df(df):
        ''' 
        formats the dataframe such that we have only mentors as columns and mentees are all the rows 
        mentees are in order of their priority
        
        the values are as follows:
        Strongest interest: 1
        Strong interest: 2
        Interested: 3
        
        returns tuple containing  
        (metadata, data)
        '''
        ## Replace statements by #s
        df = df.replace("Strongest interest", 1)
        df = df.replace("Strong interest", 2)
        df = df.replace("Interested", 3)
        
        df.index = df['Email']
        df['Priority'] = df.apply(Preferences.getGroup, axis=1)
            
        df = df.sort_values(by=['Priority', 'Timestamp'])
        df.index = df['Email']
        return (df[['Priority', 'Group', 'Year', 'Timestamp']], df.drop(columns=['Priority', 'Group', 'Year', 'Email', 'Timestamp'])) 
    
    def getPreference(self, student):
        '''
        For a given student get their preferences
        '''
        return self.df.loc[student].dropna()
    
    def getHighestPreference(self, student):
        '''
        For a given student get the rank of their highest preference
        e.g. if student ranked everyone 2 or 3 return 2
        e.g. if student ranked at least 1 person 1 return 1
        '''
        return self.getPreference(student).min()
    
    def addMentor(self, series):
        '''
        Add a mentor to the list of preferences. This is useful if a mentor was not avaliable for a previous round and is to be added to this round
        
        Args:
            series: (pd.Series) a pd Series object which contains the mentor as the name, emails as the index and preferences (in numbers) as the values
        '''
        self.df = pd.concat([series, self.df], axis=1)
    
    def dropMentor(self, mentor):
        '''
        Drop mentor from the list of preferences
        '''
        self.df.drop(columns=[mentor], inplace=True)
    
    def getNumberMentorsRanked(self):
        '''
        Get the number of mentors a student ranks ideally a student should be ranking 4 mentors
        '''
        return self.df.sum(axis=1)
    
    
    def removePreferences(self, mentor, students):
        '''
        Manually remove preferences of students for a particular mentor. This is useful if preferences can not be removed automatically (e.g. combined in         table and now not combined
        '''
        for i in students:
            try:
                self.df.loc[i, mentor] = np.nan 
            except KeyError:
                print("WARNING: {} not changed".format(i))
        
                  
class Matching():
    ''' This object performs a round of matching '''
    def __init__(self, preference, max_per_group=2, hard_max_per_group=10, min_per_group=2, rank_1_pts=10, rank_2_pts=5, rank_3_pts=1, rank_1_description="Strongest Interest", rank_2_description="Strong Interest", rank_3_description="Interested", manual_match_mentor=[]):
        ''' Initialize the object
            Matching --> dictionary of the computed optimal matching
            
        '''
        # data
        self.preference = preference.df
        self.preferenceMeta = preference.metaData
        self.matching = { i:[] for i in preference.df.columns }
        
        # matching parameters
        self.MAX_PER_GROUP = max_per_group
        self.HARD_MAX_PER_GROUP = hard_max_per_group
        self.MIN_PER_GROUP = min_per_group # give a warning when less than this in a group
        
        # mentor popularity parameters 
        self.rank_1_pts = rank_1_pts
        self.rank_2_pts = rank_2_pts
        self.rank_3_pts = rank_3_pts
        self.rank_1_description = rank_1_description
        self.rank_2_description = rank_2_description
        self.rank_3_description = rank_3_description
        
        # get mentor popularity
        self.mentor_popularity = self.getMentorPopularity()
        self.matching_performed = False # whether matching has been performed
        
        # manual match mentor 
        self.manual_match_mentor = manual_match_mentor
        
    
    def combineMentors(self, mentor1, mentor2):
        ''' This combines the mentors into one column '''
        self.preference[mentor1 + ' ' + mentor2] = self.preference.apply(lambda x: Matching._combineMentorsHelper(x[mentor1], x[mentor2]), axis=1)
        self.preference = self.preference.drop(columns=[mentor1, mentor2])
        
        # also combine their popularity
        self.mentor_popularity.loc[mentor1 + ' ' + mentor2] = self.mentor_popularity.loc[mentor1] + self.mentor_popularity.loc[mentor2]
        
        self.mentor_popularity = self.mentor_popularity.drop([mentor1, mentor2])
        
        # add this new group 
        self.matching[mentor1 + ' ' + mentor2] = []
        self.matching.pop(mentor1)
        self.matching.pop(mentor2)
    
    @staticmethod
    def _combineMentorsHelper(choice1, choice2):
        if choice1 is np.nan:
            return choice2 
        elif choice2 is np.nan:
            return choice1
        elif choice1 < choice2:
            return choice1
        else:
            return choice2
        
    
    def getParameters(self):
        ''' This prints the parameters '''
        print("Mentor Popularity Parameters:")
        print(f"{self.rank_1_description} points: {self.rank_1_pts}")
        print(f"{self.rank_2_description} points: {self.rank_2_pts}")
        print(f"{self.rank_3_description} points: {self.rank_3_pts}")
        
        print("Matching Parameters")
        print(f'\tMax Per Group: {self.MAX_PER_GROUP}')
        print(f'\tHard Max Per Group: {self.HARD_MAX_PER_GROUP}')
        print(f'\tMax Per Group: {self.MIN_PER_GROUP}')
        
    
    def getAlgorithmDescription(self):
        ''' prints a description of the algorithm using the parameters '''
        print("Part A: Determine Mentor Popularity")
        print("\t1. Rank mentors based on their popularity based on how many students ranked them and how strongly students ranked them criteria:")
        print(f"\t\tFor each Student that has {self.rank_1_description} award {self.rank_1_pts}")
        print(f"\t\tFor each Student that has {self.rank_2_description} award {self.rank_2_pts}")
        print(f"\t\tFor each Student that has {self.rank_3_description} award {self.rank_3_pts}")
        
        print("Part B: Pair each student with a mentorship group")
        print(f'\t1. Sort students based on their priority students graduating soon have priority over lower year students')
        print(f'\t2. For Each Student: (From highest to lowest priority)')
        print(f'\t\t2.1 Sort preferences by rank (highest to lowest) and mentor popularity (lowest to highest)')
        print(f"\t\t2.2 Try to pair the student with their least popular highest ranked mentor. Pairing is successful if there are {self.MAX_PER_GROUP} or less students in the mentor's group")
        print(f'\t\t2.3 If pairing is unsucessful, go down the students list of mentors untill we are successful')
        print(f'\t\t2.4 If pairing is unsucessful repeat steps 2.1-2.3 with less stringent threshold of {self.HARD_MAX_PER_GROUP}') 
        
        
    def getMentorPopularity(self):
        ''' 
        checks the mentor popularity, returns a dataframe of mentors ranked from least to most popular and their rank
        1 = gets 10 points
        2 = gets 5 points
        3 = gets 1 point
        ''' 
        tmp = self.preference.melt(value_name='RANK', var_name='MENTOR' ).dropna()
        tmp.loc[tmp['RANK'] == 1, 'RANK'] = self.rank_1_pts # strongest interested
        tmp.loc[tmp['RANK'] == 2, 'RANK'] = self.rank_2_pts # strong interest 
        tmp.loc[tmp['RANK'] == 3, 'RANK'] = self.rank_3_pts # interested
        return tmp.groupby('MENTOR').sum().sort_values(by='RANK', ascending=True)['RANK']
    
    
    def checkMatches(self):
        ''' 
        runs all sanity checks 
        1. Mentees only assigned to one group
        1. All groups are greater than min
        2. No group is greater than max
        check some sanity checks on the matches '''
        if not self.matching_performed:
            raise Exception("Please run match() method first")
        print("=" * 10 + "STARTING TEST checkGreaterThanMin() " + "=" *10)
        count = self.checkGreaterThanMin()
        print("In total {}/{} groups were less than min of {}".format(count, len(self.preference.columns), self.MIN_PER_GROUP))
        
        
        print("=" * 10 + "STARTING TEST checkLessThanMax() " + "=" *10)
        count = self.checkLessThanMax()
        print("In total {}/{} groups were greater than max of {}".format(count, len(self.preference.columns), self.MAX_PER_GROUP))
        
        print("=" * 10 + "STARTING TEST checkMentorGroupsSound() " + "=" *10)
        count = self.checkMentorGroupsSound()

        
        print("=" * 10 + "STARTING TEST checkMenteesSound() " + "=" *10)
        count = self.checkMenteesSound()
        
        
    def checkGreaterThanMin(self):
        matching_df = self.getMatchingDf()
        counts = 0
        for i in matching_df.keys():
            if len(matching_df[i].dropna()) < self.MIN_PER_GROUP:
                print("WARNING: {} only contains {} members".format(i, len(matching_df[i].dropna())))
                counts += 1
        return counts
    
    def checkLessThanMax(self):
        '''
        prints warning message for gorups over the max_per_group.
        Returns the count of groups that achieve this
        '''
        matching_df = self.getMatchingDf()
        counts = 0
        for i in matching_df.keys():
            if len(matching_df[i].dropna()) > self.HARD_MAX_PER_GROUP:
                raise Exception("ERROR: {} contains {} members - over hard maximum of {}".format(i, len(matching_df[i].dropna()), self.HARD_MAX_PER_GROUP))
                
            if len(matching_df[i].dropna()) > self.MAX_PER_GROUP:
                print("WARNING: {} contains {} members".format(i, len(matching_df[i].dropna())))
                counts += 1
        return counts
    
            
    def checkMentorGroupsSound(self):
        ''' 
        check that a duplicate name does not once in a group (e.g. a group has two of the same person
        '''
        matching_df = self.getMatchingDf()
        for i in matching_df.keys():
            if (len(matching_df[i].dropna().drop_duplicates()) != len(matching_df[i].dropna())):
                raise Exception("ERROR: {} is inconsistent, a name appears twice".format(i))
            
    
    def checkMenteesSound(self):
        '''
        Ensure that a single mentee is not assigned to more than one group (in a single rotation) and no mentee left out
        '''
        if len(self.getMeltedMatches()) == len(self.preference.index):
            return
        else:
            raise Exception("ERROR: There are duplicate mentees, one mentee may be assigned to more than one mentor")
    
    
    def checkProportionPreference(self):
        '''
        Checks the proportion of the participants that got rank 1, rank 2 and rank 3 and prints to output
        '''
        meltedMatches = self.getMeltedMatches()
        ranks = { 1:0, 2:0, 3:0 }
        no_rank_1 = 0
        no_rank_1_2 = 0
        for mentee, participant_preferences in self.preference.iterrows():
            mentee_match = meltedMatches[mentee]
            mentee_ranks = participant_preferences.dropna()
            
            rank_of_matched = mentee_ranks[mentee_match] # the rank of the
            
            ranks[rank_of_matched] += 1
            
            #print(mentee_ranks)
            
            if len(mentee_ranks[mentee_ranks == 1]) == 0:
                #print('no rank 1')
                no_rank_1 += 1
            if len(mentee_ranks[(1 <= mentee_ranks) & (mentee_ranks <= 2)]) == 0:
                #print('no rank 1 or 2')
                no_rank_1_2 += 1
            #print()
        
        print("Out of {} mentees".format(len(meltedMatches)))
        print("{} ({:.0%}) were matched with their strongest interest".format(ranks[1], ranks[1] / len(meltedMatches)))
        print("{} ({:.0%}) were matched with their strong interest".format(ranks[2], ranks[2] / len(meltedMatches)))
        print("{} ({:.0%}) were matched with their interested".format(ranks[3], ranks[3] / len(meltedMatches)))
        print("{} ({:.0%}) mentees did not have a strongest interest preference".format(ranks[3], ranks[3] / len(meltedMatches)))
        print("{} ({:.0%}) mentees only had an interested preference".format(ranks[3], ranks[3] / len(meltedMatches)))
        
    
    def getMeltedMatches(self):
        ''' 
        get a df where have student as index and mentor is the value
        '''
        tmp = self.getMatchingDf().melt(var_name='MENTOR', value_name='STUDENT').dropna()
        matching_melt = tmp['MENTOR']
        matching_melt.index = tmp['STUDENT']
        return matching_melt 
    
    def getMatchingDf(self):           
        ''' Get a dataframe where the columns are mentors and each row is a student. Have to pad with NaN values for this to work '''
        ## compute max group 
        max_persons_in_grp = max([len(i) for i in self.matching.values()])

        matching_out = self.matching
        for i in matching_out.keys():
            while len(matching_out[i]) < max_persons_in_grp:
                matching_out[i].append(np.nan)
        return pd.DataFrame(matching_out)
    
    def getGroupSizes(self):
        ''' Get a pandas series where we record group sizes for each mentorship group '''
        if not self.matching_performed:
            raise Exception("Please run match() method first")
        return self.getMeltedMatches().reset_index().groupby("MENTOR").count()['STUDENT']
    
    def getPopularityGroupSize(self):
        ''' get a dataframe which has the group sizes and the popularity and the student's rank '''
        tmp = pd.concat([self.getGroupSizes(), self.mentor_popularity], axis=1)
        tmp.columns = ['GROUP_SIZE', 'POPULARITY']
 
        return tmp.sort_values(by='POPULARITY', ascending=False).fillna(0)

    
    def getStudentPreferencesMatches(self):
        ''' for each student get the preference of the group they were matched with '''
        tmp = self.getMeltedMatches().reset_index()
        preferences = self.preference.reset_index().melt(id_vars=['Email'], value_name='RANK', var_name='MENTOR').dropna().reset_index()
        return tmp.merge(preferences, left_on=['STUDENT', 'MENTOR'], right_on=['Email', 'MENTOR']).drop(columns=['index', 'STUDENT'])

    
    def plotStudentPreferences(self):
        bins = np.arange(1,5,1)
        plt.hist(self.getStudentPreferencesMatches()['RANK'], np.arange(1,5,1), align='left')
        plt.xticks(bins[:-1])
        plt.xlabel("Rank")
        plt.ylabel("Count")
        return
    
    def plotMentorPopularityGroupSize(self):
        tmp = self.getPopularityGroupSize().reset_index()
        tmp['MENTOR_POPULARITY'] = tmp['MENTOR'] + " Popularity = " + tmp['POPULARITY'].astype(int).astype(str)
        return sns.barplot(y='MENTOR_POPULARITY', x='GROUP_SIZE', data=tmp)
        pass
    
    def manualMatch(self, student_preference):
        print(student_preference)
    
    def pair(self, student_preference, max_per_group):
        ''' 
        pairs the participant with a group 
        3 parameters are taken into account:
            1. Rank of participant
            2. Popularity of mentor 
            3. number of people in the group

        General algorithm: For each "rank" of the participant, try to match the participant with the least popular mentor. If we are over the expected group size than 
        try the next least popular mentor ..
        
        student_preference - preferences of student, 1 being the highest
        '''
        
        for rank_level in [1,2,3]: # right now there are 3 ranks

            ## get mentors in order of lowest popularity
            possible_mentors = (self.mentor_popularity.loc[student_preference[student_preference == rank_level].index.values].sort_values(ascending=True).index.values)
            
            ##### check if we should manually match
            for i in self.manual_match_mentor:
                if i in possible_mentors:
                    return i
                                
            if (len(possible_mentors) > 1):
                assert(self.mentor_popularity.loc[possible_mentors[0]] <= self.mentor_popularity.loc[possible_mentors[1]])
            
            ## try to match with group (in order of lowest popularity
            for m in possible_mentors:
                if len(self.matching[m]) < max_per_group:
                    return m
        
        return None
        
            
    def match(self):
        ''' given a preferences dataframe (name as index, preferences other columns), computes the optimal matches '''
        for name, participant_preferences in self.preference.iterrows():
            ranks = participant_preferences.dropna().sort_values()
            needsMatch = True

            ## try pairing with the initial threshold
            matched_mentor = self.pair(participant_preferences, self.MAX_PER_GROUP)
            if matched_mentor is not None:
                self.matching[matched_mentor].append(name)
                needsMatch = False

            ## if could not pair with initial threshold, then try to pair with more loose threshold
            if needsMatch:
                
                ## try pairing with the initial threshold
                matched_mentor = self.pair(participant_preferences, self.HARD_MAX_PER_GROUP)
                #print("Second round matched mentor: {}".format(matched_mentor))
                if matched_mentor is not None:
                    self.matching[matched_mentor].append(name)
                    #print("Paired {} with {}!".format(name, matched_mentor))
                    needsMatch = False
                
            # If still could not pair, throw an error with this student
            if needsMatch:
                print("ERROR: {} could not be paired".format(name))
                print(ranks)
                print("")
        self.matching_performed = True
                
    
    def filterMatchesFromPreference(self):
        ''' mark the matches for the previous roundtable as NaN for preparation for next round'''
        matching_melt = self.getMeltedMatches()
        preferences_new = self.preference.copy()
        for name, student_preference in preferences_new.iterrows():
            try:
                student_preference[matching_melt[name]] = np.nan
            except KeyError:
                print("WARNING: {} not changed".format(name))
        out = Preferences(preferences_new, metaData=self.preferenceMeta) 
        return out

            
class FinalPairing:
    ''' 
    This class is for summarizing the final pairing
    '''
    def __init__(self, arr_of_matches, initial_preferences):
        self.df = FinalPairing.combineMatches(arr_of_matches)
        self.matches = arr_of_matches
        self.preference = initial_preferences

    @staticmethod
    def combineMatches(matches):
        '''
        Combine matches into one dataframe, each row is a student and each column is their paired group
        matches array of match objects
        '''

        match_dfs = [ m.getMeltedMatches() for m in matches ]
        out =  pd.concat(match_dfs, axis=1)
        out.columns = [ "MENTOR_{}".format(i + 1) for i in range(len(matches)) ] # rename columns 
        return out
        
    def checkProportionTop(self):
        '''
        check the proportion of students that have at least 1 top choice, none of their top choice but at least 1 second choice, or only 3rd choices
        '''
        tmpArr = []
        for i in self.matches:
            tmp = i.getStudentPreferencesMatches()
            tmp.index= tmp['Email']
            tmpArr.append(tmp['RANK'])

        student_ranks = pd.concat(tmpArr, axis=1).min(axis=1) # array with rank mentor was rated for all matches
            
        print("Out of {} mentees".format(len(student_ranks)))
        rank_1 = student_ranks[student_ranks == 1]
        rank_2 = student_ranks[student_ranks == 2] # students that achieve max of rank 2
        rank_3 = student_ranks[student_ranks == 3] # students that achieve max of rank 3
        print("{} ({:.0%}) were matched with at least 1 of their strongest interest".format(len(rank_1), len(rank_1) / len(student_ranks)))
        print("{} ({:.0%}) were matched with at least 1 of their strong interest (no strongest interest)".format(len(rank_2), len(rank_2) / len(student_ranks)))
        print("{} ({:.0%}) were not matched with any of their strongest/strong interest".format(len(rank_3), len(rank_3) / len(student_ranks)))
        
        
        # see if it is even possible to pair with strongest interest (do these exist)
        
        rank2_3 = pd.concat([rank_2, rank_3])
        
        highest_preference = pd.Series(rank2_3.index).apply(self.preference.getHighestPreference)
        highest_preference_is_2 = highest_preference[highest_preference == 2]
        highest_preference_is_3 = highest_preference[highest_preference == 3]
        print("{} ({:.0%}) students did not have a strongest interest".format(len(highest_preference_is_2), len(highest_preference_is_2) / len(student_ranks)))
        print("{} ({:.0%}) students did not have a strongest or strong interest".format(len(highest_preference_is_3), len(highest_preference_is_3) / len(student_ranks)))
        return student_ranks

        
        
        
        
