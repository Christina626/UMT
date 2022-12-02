class GlobalVariable:
  _global_dict={}
  def __init__(self):
    global _global_dict
    _global_dict={}
  def set_value(self,key,value):
   _global_dict[key]=value
  def get_value(self,key,dftvalue=None):
    try: 
        return _global_dict[key]
    except KeyError:
        return dftvalue
glob=GlobalVariable()
    
