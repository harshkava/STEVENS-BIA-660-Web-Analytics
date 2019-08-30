# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:01:53 2018

@author: Harsh Kava
"""

a =[1,2,3,3,4,5,6]
a[:2]



for actor in actorSet:
                    print(actor)
                    if actor in actorlist:
                        if actor in actor_likes:
                            print('Match in FB List')
                            actorScores.append(actor_likes[actor])
                        else:
                            print('Not in FB List')
                            actorScores.append(0)
                    else:
                        print('Not Present')
                        actorScores.append(0)