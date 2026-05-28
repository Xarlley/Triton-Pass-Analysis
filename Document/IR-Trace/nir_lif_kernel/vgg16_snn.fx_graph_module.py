# nirtorch.nir_to_torch 重建的 fx.GraphModule.forward 源码（真实 dump）




def forward(self, input, state : typing_Dict[str,typing_Any] = None):
    ones = torch.ones(1);  ones = None
    input_1 = input
    is_none = _operator_is_(state, None)
    initialized_state = nirtorch_nir_interpreter_ternary_operator(is_none, {'input_1': None, '_0_0': None, '_0_2': None, '_0_3': None, '_0_5': None, '_0_6': None, '_0_7': None, '_0_9': None, '_0_10': None, '_0_12': None, '_0_13': None, '_0_14': None, '_0_16': None, '_0_17': None, '_0_19': None, '_0_20': None, '_0_22': None, '_0_23': None, '_0_24': None, '_0_26': None, '_0_27': None, '_0_29': None, '_0_30': None, '_0_32': None, '_0_33': None, '_0_34': None, '_0_36': None, '_0_37': None, '_0_39': None, '_0_40': None, '_0_42': None, '_0_43': None, '_1_0': None, '_1_1': None, '_1_2': None, '_1_3': None, '_1_4': None, '_1_5': None, 'output': None}, state);  is_none = state = None
    _0_0 = self._0_0(input_1);  input_1 = None
    _0_2 = self._0_2(_0_0);  _0_0 = None
    _0_3 = self._0_3(_0_2);  _0_2 = None
    _0_5 = self._0_5(_0_3);  _0_3 = None
    _0_6 = self._0_6(_0_5);  _0_5 = None
    _0_7 = self._0_7(_0_6);  _0_6 = None
    _0_9 = self._0_9(_0_7);  _0_7 = None
    _0_10 = self._0_10(_0_9);  _0_9 = None
    _0_12 = self._0_12(_0_10);  _0_10 = None
    _0_13 = self._0_13(_0_12);  _0_12 = None
    _0_14 = self._0_14(_0_13);  _0_13 = None
    _0_16 = self._0_16(_0_14);  _0_14 = None
    _0_17 = self._0_17(_0_16);  _0_16 = None
    _0_19 = self._0_19(_0_17);  _0_17 = None
    _0_20 = self._0_20(_0_19);  _0_19 = None
    _0_22 = self._0_22(_0_20);  _0_20 = None
    _0_23 = self._0_23(_0_22);  _0_22 = None
    _0_24 = self._0_24(_0_23);  _0_23 = None
    _0_26 = self._0_26(_0_24);  _0_24 = None
    _0_27 = self._0_27(_0_26);  _0_26 = None
    _0_29 = self._0_29(_0_27);  _0_27 = None
    _0_30 = self._0_30(_0_29);  _0_29 = None
    _0_32 = self._0_32(_0_30);  _0_30 = None
    _0_33 = self._0_33(_0_32);  _0_32 = None
    _0_34 = self._0_34(_0_33);  _0_33 = None
    _0_36 = self._0_36(_0_34);  _0_34 = None
    _0_37 = self._0_37(_0_36);  _0_36 = None
    _0_39 = self._0_39(_0_37);  _0_37 = None
    _0_40 = self._0_40(_0_39);  _0_39 = None
    _0_42 = self._0_42(_0_40);  _0_40 = None
    _0_43 = self._0_43(_0_42);  _0_42 = None
    _1_0 = self._1_0(_0_43);  _0_43 = None
    _1_1 = self._1_1(_1_0);  _1_0 = None
    _1_2 = self._1_2(_1_1);  _1_1 = None
    _1_3 = self._1_3(_1_2);  _1_2 = None
    _1_4 = self._1_4(_1_3);  _1_3 = None
    _1_5 = self._1_5(_1_4);  _1_4 = None
    return (_1_5, initialized_state)
    