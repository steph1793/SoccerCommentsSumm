from pyrouge import Rouge155

r=Rouge155()
scores = r.get_scores([], [], avg=True)