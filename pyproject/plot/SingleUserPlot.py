import numpy as np

from pyproject.plot.SingleUserTotalData import plotSingleUser

if __name__ == '__main__':
	random_integers = np.random.randint(1000, 7444, 200)
	for user in random_integers:
		plotSingleUser(user, 'D:/paper_result/200_random_single_user')
	