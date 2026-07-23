<H3>Name:Ramya P</H3>
<H3>Register No:212223230168</H3>
<H3>Date:23-7-26</H3>
<h1 align =center>Implementation of Exact Inference Method of Bayesian Network</h1>

## Aim:
To implement the inference Burglary P(B| j,⥗m) in alarm problem by using Variable Elimination method in Python.

## Algorithm:

Step 1: Define the Bayesian Network structure for alarm problem with 5 random variables, Burglary,Earthquake,John Call,Mary Call and Alarm.<br>
Step 2: Define the Conditional Probability Distributions (CPDs) for each variable using the TabularCPD class from the pgmpy library.<br>
Step 3: Add the CPDs to the network.<br>
Step 4: Initialize the inference engine using the VariableElimination class from the pgmpy library.<br>
Step 5: Define the evidence (observed variables) and query variables.<br>
Step 6: Perform exact inference using the defined evidence and query variables.<br>
Step 7: Print the results.<br>

## Program :
```
# Import required libraries
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define Bayesian Network structure
network = DiscreteBayesianNetwork([
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
])

# Define the conditional probability distributions
cpd_burglary = TabularCPD(
    variable='Burglary',
    variable_card=2,
    values=[[0.999], [0.001]]
)

cpd_earthquake = TabularCPD(
    variable='Earthquake',
    variable_card=2,
    values=[[0.998], [0.002]]
)

cpd_alarm = TabularCPD(
    variable='Alarm',
    variable_card=2,
    values=[
        [0.999, 0.71, 0.06, 0.05],
        [0.001, 0.29, 0.94, 0.95]
    ],
    evidence=['Burglary', 'Earthquake'],
    evidence_card=[2, 2]
)

cpd_john_calls = TabularCPD(
    variable='JohnCalls',
    variable_card=2,
    values=[
        [0.95, 0.1],
        [0.05, 0.9]
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

cpd_mary_calls = TabularCPD(
    variable='MaryCalls',
    variable_card=2,
    values=[
        [0.99, 0.3],
        [0.01, 0.7]
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

# Add CPDs to the network
network.add_cpds(
    cpd_burglary,
    cpd_earthquake,
    cpd_alarm,
    cpd_john_calls,
    cpd_mary_calls
)

# Check whether the model is valid
print(network.check_model())

# Initialize inference engine
inference = VariableElimination(network)

# Exact inference - 1
evidence = {
    'JohnCalls': 1,
    'MaryCalls': 0
}

result = inference.query(
    variables=['Burglary'],
    evidence=evidence
)

print("When John calls and Mary does not call:")
print(result)

# Exact inference - 2
evidence1 = {
    'JohnCalls': 1,
    'MaryCalls': 1
}

result2 = inference.query(
    variables=['Burglary'],
    evidence=evidence1
)

print("When both John and Mary call:")
print(result2)
```


## Output :
<img width="697" height="416" alt="image" src="https://github.com/user-attachments/assets/347f505b-49dc-4d7d-8302-e812da8c5dc2" />


## Result :
Thus, Bayesian Inference was successfully determined using Variable Elimination Method

