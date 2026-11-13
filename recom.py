import json


def recom(CORE):
    
    if 'top' not in CORE.req:   
        top = CORE.config.top
    else:
        try:
            top = int(CORE.req['top'])
        except:
            answer = {'status': 'err', 'message' :'Bad secret "top" key'}
            return json.dumps(answer)

    if top < 1 or top > 100:
        top = CORE.config.top

    user_id = CORE.req['user_id']

    chanells = CORE.model.predict(user_id)  
    answer = {'status': 'ok', 'user_id': user_id, 'recom': chanells[0:top]}

    return json.dumps(answer)