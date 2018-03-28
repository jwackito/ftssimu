from rucio.core.rse import get_rse_id
from rucio.extensions.forecast import *
import json
import datetime
import pandas as pd
from time import sleep

@read_session
def get_session(session=None):
    return session

session = get_session()

m = T3CModel()

src = 'AGLT2_DATADISK'
dst = 'MWT2_DATADISK'

site_src = m.rse2site(src)
site_dst = m.rse2site(dst)
rses_id_src = m.site2rseids(m.rse2site(src))
rses_id_dst = m.site2rseids(m.rse2site(dst))


def get_frames():
    video = session.query(models.Request).filter(models.Request.state == models.RequestState.SUBMITTED,
                                           models.Request.source_rse_id.in_(rses_id_src),
                                           models.Request.dest_rse_id.in_(rses_id_dst)
                                          ).order_by(models.Request.submitted_at).all()
    return video

def get_transfers(span=7):
    '''get the transfers created during the last 7 days'''
    req = models.Request.__history_mapper__.class_
    q = session.query(req).filter(req.created_at > datetime.datetime.utcnow()-datetime.timedelta(days=span),
            req.state == models.RequestState.DONE,
            req.source_rse_id.in_(rses_id_src),
            req.dest_rse_id.in_(rses_id_dst)
            ).order_by(req.submitted_at).all()
    return q

def process(f):
    f.pop('_sa_instance_state')
    f.pop('attributes')
    f.pop('did_type')
    f.pop('state')
    f.pop('request_type')
    f.pop('adler32')
    f.pop('md5')
    f.pop('err_msg')
    f.pop('estimated_at')
    f.pop('estimated_started_at')
    f.pop('estimated_transferred_at')
    f.pop('previous_attempt_id')
    f.pop('requested_at')
    return f

print 'Getting transfers'
data = get_transfers(7)
print len(data), 'transfers recovered'
environments = []
i = 0
for transfer in data:
    pd.DataFrame([process(transfer.to_dict())]).to_csv('%05d.csv'%(i), index=False)
    i += 1

