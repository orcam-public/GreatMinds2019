from __future__ import print_function

import json
import numpy as np
from urllib.request import Request
from urllib.request import urlopen


SERVER_URL = 'http://orcam-datahack-server.herokuapp.com/'


def evaluate(submission, test_labels, verbose=True):
    # Find where on the submission predicted index equals to the gt
    hits = np.array(submission, dtype='int') == test_labels[:, np.newaxis]
    # Compute and display top 1 / 5 accuracies
    top1_accuracy = np.mean(hits[:, 0]) * 100
    top5_accuracy = np.mean(np.any(hits, axis=1)) * 100
    score = np.mean(np.sum(np.arange(100, 50, -10)[np.newaxis] * hits, axis=1))

    if verbose:
        print('top 1 accuracy {:.2f}%'.format(top1_accuracy))
        print('top 5 accuracy {:.2f}%'.format(top5_accuracy))
        print('mean score: {:.2f}'.format(score))
    return top1_accuracy, top5_accuracy


def submit(name, submission):
    # Submit your result to the leaderboard
    jsonStr = json.dumps({'submitter': name, 'predictions': submission})
    data = jsonStr.encode('utf-8')
    req = Request(SERVER_URL,
                  headers={'Content-Type': 'application/json'},
                  data=data)
    resp = urlopen(req)
    print(json.load(resp))
