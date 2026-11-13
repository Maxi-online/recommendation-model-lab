import json


def popular(CORE):
    
    answer = {'status': 'ok', 'popular': CORE.model.popular_list}

    return json.dumps(answer)