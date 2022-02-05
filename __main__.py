from machine_learning_modelling import *
from parameter_computation import *

def authentication(predictions,user_predictions,attacker_predictions,a,b,seg):

 
    steps=[0]*(math.ceil(len(attacker_predictions)/seg))
    appr = [3*i +1 for i in range(math.ceil((seg)/3))]
    sessions = {k : [] for k in range(math.floor(len(attacker_predictions)/seg))}
    
    for i in range(len(attacker_predictions)-seg):
        if i>=math.floor(len(attacker_predictions)/seg):
            break
        k=seg*i
        for j in range(k,k+seg):
            sessions[i].append(attacker_predictions[j])
    

    appr = set(appr)
    # Decision sequence
    D = [0.0 for _ in range(len(attacker_predictions)+1)] 
    # Initial value
    D[0] = 0.55
    for i in sessions.keys():
        for j in range(len(sessions[i])):
            if (sessions[i][j]==-1):
                D[j+1] = D[j] - pow(2,-0.005*(j))*a
                if (j) in appr:
                    if D[j+1]<D[j] and D[j]<D[j-1]:
                        D[j+1]=D[j+1]+exp(-j+1)*(b/2)
            else:
                D[j+1] = D[j] + pow(2,-0.005*(j))*b
                if (j) in appr:
                    if D[j+1]>D[j] and D[j]>D[j-1]:
                        D[j+1]=D[j+1]-exp(-j+1)*(a/2)
            
            if D[j+1] > 0.8:
                break
            
            if D[j+1] < 0.3:
                step=j+1
                steps.append(step)
                break
                

    
      
    
    steps2=[0]*(math.ceil(len(user_predictions)/seg))
    sessions = {k : [] for k in range(math.floor(len(user_predictions)/seg))}
    
    for i in range(len(user_predictions)-seg):
        if i>=math.floor(len(user_predictions)/seg):
            break
        k=seg*i
        for j in range(k,k+seg):
            sessions[i].append(user_predictions[j])
    # Decision sequence
    D = [0.0 for _ in range(len(user_predictions)+1)] 
    # Initial value
    D[0] = 0.55
    
    for i in sessions.keys():
        for j in range(len(sessions[i])):
            if (sessions[i][j]==-1):
                D[j+1] = D[j] - pow(2,-0.005*(j))*a
                if (j) in appr:
                    if D[j+1]<D[j] and D[j]<D[j-1]:
                        D[j+1]=D[j+1]+exp(-j+1)*(b/2)
            else:
                D[j+1] = D[j] + pow(2,-0.005*(j))*b
                if (j) in appr:
                    if D[j+1]>D[j] and D[j]>D[j-1]:
                        D[j+1]=D[j+1]-exp(-j+1)*(a/2)
            
            if D[j+1] > 0.8:
                step=j+1
                steps2.append(step)
                break
            
            if D[j+1] < 0.3:
                
                break
    
            
    
    steps_user=np.array(steps2)
    steps_attacker=np.array(steps)
    steps_user=steps_user[steps_user!=0]
    steps_attacker=steps_attacker[steps_attacker!=0]
    
    return steps_attacker,steps_user
                                                                            
if __name__ == '__main__':
    classifier='LOF'
    X_train_user,X_train_attacker,X_test,y_test=load_user_data(1)
    predictions,user_predictions,attacker_predictions=classification(X_train_user,X_train_attacker,X_test,y_test,classifier)
    FAR,FRR=metrics(predictions,y_test)
    k=4
    a,b=param(FAR,FRR,k)
    seg=20
    steps_attacker,steps_user=authentication(predictions,user_predictions,attacker_predictions,a,b,seg)
    user_success=len(steps_user)/math.floor(len(user_predictions)/seg)
    attacker_success=len(steps_attacker)/math.floor(len(attacker_predictions)/seg)
    mean_attacker_steps=np.mean(steps_attacker)
    mean_user_steps=np.mean(steps_user)
    print("User success: ", user_success)
    print("Attacker success: ",attacker_success)
    print("User Steps: ",mean_user_steps)
    print("Attacker Steps: ",mean_attacker_steps)
    print("FAR: ",FAR)
    print("FRR: ",FRR)
    print("Exps User: ",math.floor(len(user_predictions)/seg))
    print("Exps Attacker: ",math.floor(len(attacker_predictions)/seg))
    plt.hist(steps_attacker,rwidth=0.6,color="black")
    plt.title('Distribution of attackers swipes')
    plt.xlabel('Number of swipes')
    plt.ylabel('Number of Attackers')
    labels= ["(FAR,FRR)=(0.12,0.12)"]
    plt.legend(labels)
    fre6=list(steps_attacker).count(3)+list(steps_attacker).count(4)+list(steps_attacker).count(5)+list(steps_attacker).count(6)
    fre79=list(steps_attacker).count(7)+list(steps_attacker).count(8)+list(steps_attacker).count(9)
    frerest=len(steps_attacker)-fre79-fre6
    print("Frequncy 6: ",fre6)
    print("Frequency 79: ",fre79)
    print("Frequency Rest: ",frerest)