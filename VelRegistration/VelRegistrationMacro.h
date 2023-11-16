#pragma once

#define VelSetMacro(name, type)     \
  virtual void Set##name(type _arg) \
  {                                 \
    if (this->m_##name != _arg)     \
    {                               \
      this->m_##name = _arg;        \
    }                               \
  }

#define VelGetMacro(name, type) \
  virtual type Get##name()      \
  {                             \
    return this->m_##name;      \
  }

#define VelSetPointerMacro(name, type) \
  virtual void Set##name(type* _arg)   \
  {                                    \
    if (this->m_##name != _arg)        \
    {                                  \
      this->m_##name = _arg;           \
    }                                  \
  }

#define VelGetPointerMacro(name, type) \
  virtual type* Get##name()            \
  {                                    \
    return this->m_##name;             \
  }

#define VelSetVector3Macro(name, type)                                                                \
  virtual void Set##name(type _arg1, type _arg2, type _arg3)                                          \
  {                                                                                                   \
    if ((this->m_##name[0] != _arg1) || (this->m_##name[1] != _arg2) || (this->m_##name[2] != _arg3)) \
    {                                                                                                 \
      this->m_##name[0] = _arg1;                                                                      \
      this->m_##name[1] = _arg2;                                                                      \
      this->m_##name[2] = _arg3;                                                                      \
    }                                                                                                 \
  }                                                                                                   \
  virtual void Set##name(type _arg[3])                                                                \
  {                                                                                                   \
    this->Set##name(_arg[0], _arg[1], _arg[2]);                                                       \
  }

#define VelGetVector3Macro(name, type)                          \
  virtual type* Get##name()                                     \
  {                                                             \
    return this->m_##name;                                      \
  }                                                             \
  virtual void Get##name(type& _arg1, type& _arg2, type& _arg3) \
  {                                                             \
    _arg1 = this->m_##name[0];                                  \
    _arg2 = this->m_##name[1];                                  \
    _arg3 = this->m_##name[2];                                  \
  }                                                             \
  virtual void Get##name(type _arg[3])                          \
  {                                                             \
    this->Get##name(_arg[0], _arg[1], _arg[2]);                 \
  }
