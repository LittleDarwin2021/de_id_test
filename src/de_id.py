import pandas as pd
import streamlit as st
from scipy.stats import entropy


class De_id:
    #ユニークID(削除対象,list),順識別子（加工対象,list）,カテゴリ列(set), 機微情報（維持,str）
    def __init__(self, dataframe, unique_id, quasi_identifier, categorical, sensitive_attributes):
        self.dataframe = dataframe.drop(labels= unique_id, axis = 1)#ユニークIDを削除する
        #データ型がobjectになっているものを"category"に修正する
        for i in categorical:
          self.dataframe["{}".format(i)] = self.dataframe["{}".format(i)].astype("category")
        self.mondrian = Mondrian(self.dataframe, quasi_identifier, sensitive_attributes)
        
    def k_anonymize(self, k):
        partitions = self.mondrian.partition(k)
        return anonymize(
            self.mondrian.dataframe,
            partitions,
            self.mondrian.quasi_identifier,
            self.mondrian.sensitive_attributes,
        )


#Mondrianアルゴリズム
class Mondrian:
    def __init__(self, dataframe, quasi_identifier, sensitive_attributes = None):
        self.dataframe = dataframe
        self.quasi_identifier = quasi_identifier
        self.sensitive_attributes = sensitive_attributes

    def go_nogo(self, partition, k=2):
        if len(partition) < k:
          return False
        else:
          return True


    #カラム毎に間隔を定める（カテゴリはユニーク値数を、数値は最大値と最小値の差を、dataframeのインデックスの数で割る）
    def get_spans(self, partition, scale = None):
        spans = {}
        #準識別子毎に実行
        for column in self.quasi_identifier:
            #その準識別子がカテゴリカラムの場合,ユニーク値の数をカウントする
            if self.dataframe[column].dtype.name in ["object","category", "str"]:
                span = len(self.dataframe[column][partition].unique())
            #数値カラムの場合,partitionの最大と最小のを引いたものをspanとする
            else:
                span = (
                    self.dataframe[column][partition].max() - self.dataframe[column][partition].min()
                )
            #間隔を定める。scaleはdataframeのインデックス数
            if scale :
                span = span / scale[column]
            #その準識別子の間隔を辞書型で保存する
            spans[column] = span
        #spansは辞書型
        return spans

    def split(self, column, partition):
        dfp = self.dataframe[column][partition]
        
        #カテゴリ値の場合はユニーク値のセットの長さを半分で割る
        if dfp.dtype.name in ["object","category", "str"]:
            values = dfp.unique()
            lv = set(values[: len(values) // 2])
            rv = set(values[len(values) // 2 :])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        #数値の場合は中央値を境界にして分割する
        else:
            median = dfp.median()
            lhs = dfp.index[dfp < median]
            rhs = dfp.index[dfp >= median]
            return (lhs, rhs)

    #k値を設定した上、それを満たすパーティションを得る。例：[(5,6), (7,8), (0,1), (2,3,4)]
    #特徴量を一つ選びユニーク値の中央値で分割する工程を、どの特徴量でもk値を満たせなくなるまで繰り返す
    def partition(self, k = 2):
        #dataframeのインデックスの間隔を得る
        scale = self.get_spans(self.dataframe.index)
        new_partitions = []
        partitions = [self.dataframe.index]
        #partisionから全てなくなるまで繰り返す
        while partitions:
            #partisionの先頭から抜き出し、全てなくなるまで処理を繰り返す
            partition = partitions.pop(0)
            #カラム毎に(partisionの範囲での最大値-最小値）/データ全体のユニーク値数で割って新たな間隔を得る
            spans = self.get_spans(partition, scale)
            #辞書spansの値をソートしカラム名を順に取得する
            #spanの値が大きい特徴量から計算を行い、kを満たせば次の特徴量にはいかずforループを抜ける
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                #データを分割する
                lp, rp = self.split(column, partition)
                #2つに分けた群が両方ともk値以上になっていればpartitionに分割区間を追加、一つでも満たしていない場合はforループで次のcolumnに
                if not self.go_nogo(lp, k) or not self.go_nogo(rp, k):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                new_partitions.append(partition)
        return new_partitions


def agg_categorical_column(series):
    series.astype("category")

    label = [str(n) for n in set(series)]
    return [",".join(label)]


def agg_numerical_column(series):
    min = series.min()
    max = series.max()
    if min == max:
        value = str(min)
    else:
        value = "{}-{}".format(min,max)
    return [value]


def anonymize(dataframe, partitions, quasi_identifier, sensitive_attributes, max_partitions = None):
    #準識別子がカテゴリか数値かで区別する
    category_or_numeric = {}
    for i in quasi_identifier:
        if dataframe[i].dtype.name in ["object","category", "str"]:
            category_or_numeric[i] = agg_categorical_column
        else:
            category_or_numeric[i] = agg_numerical_column
    
    #全ての順識別子の組み合わせの数をカウントする
    agg_data = []
    for num, partition in enumerate(partitions):
        if max_partitions is not None and num > max_partitions:
            break
        
        #python tips .aggに自作関数を指定すると集計結果（この場合Series）を関数の引数として実行させられる
        grouped_columns = dataframe.loc[partition].agg(category_or_numeric, squeeze = False)
        #機微情報がある場合
        if sensitive_attributes:
  
            sensitive_counts = (
                dataframe.loc[partition].groupby(sensitive_attributes).agg({sensitive_attributes: "count"})
            )
            values = grouped_columns.apply(lambda x: x[0]).to_dict()
            for sensitive_value, count in sensitive_counts[sensitive_attributes].items():
                if count == 0:
                    continue
                values.update(
                    {
                        sensitive_attributes: sensitive_value,
                        "count": count,
                    }
                )
                agg_data.append(values.copy())
        #機微情報がない場合
        else:
            counts = (df.loc[partition].groupby(quasi_identifier).agg({quasi_identifier[0]: "count"}) )
            counts = counts.rename(columns = {quasi_identifier[0]:"count"})
            values = grouped_columns.apply(lambda x: x[0]).to_dict()
            
            _count = 0
            for val, count in counts["count"].items():
              _count = _count + count
              if count == 0:
                continue
              values.update({"count": _count})
            agg_data.append(values.copy())
              

            
    return agg_data



#タイトル
st.title("匿名加工アプリ")
uploaded_files = st.file_uploader("匿名加工を行うcsvファイルを選択してください。サーバーにファイルは残りません。", accept_multiple_files= False)
k_value = st.number_input("K値を指定してください",min_value=2)
if uploaded_files:
    df = pd.read_csv(uploaded_files)
    
    st.write("加工前のテーブル")
    st.write(df.head())
    st.write("-----------------------------------")
    st.markdown("#### Step1 変数の選択")

    unique_id = st.multiselect("ID,氏名など個人を特定できる識別子を選択してください。識別子は匿名加工の際に削除されます。" , df.columns)
    if unique_id:
        quasi_identifier = st.multiselect("匿名加工対象となる準識別子を複数選択してください。" , df.drop(unique_id, axis = 1).columns)
    else:
        quasi_identifier = st.multiselect("匿名加工対象となる準識別子を複数選択してください。" , df.columns)
    if quasi_identifier:
        categorical = st.multiselect("準識別子のうち、カテゴリー変数を指定してください。" , quasi_identifier)
        categorical = set(categorical)
    else:
        categorical = set()
    #if unique_id and quasi_identifier:
        #sensitive_attributes = st.selectbox("センシティブ情報を一つ選択してください。こちらは最後まで加工されません。" , df.drop(unique_id, axis = 1).drop(quasi_identifier, axis = 1).columns)

    st.write("-----------------------------------")
    st.markdown("#### Step2 抽象化")
    df2 = df.copy()
    #st.session_stateにより変数を維持させることで複数回繰り返しののデータフレーム修正を可能にする
    #https://docs.streamlit.io/library/advanced-features/session-state#initialization'
    if categorical:
        to_modify = st.selectbox("抽象化を行うカラムを選択してください" , categorical)
        st.write(to_modify)
        modify_data = st.multiselect("抽象化を行うデータを選択してください（複数可）" , df[to_modify].unique())
        modify_word = st.text_input("何という言葉に置き換えますか")
        execute = st.button("抽象化を実行する")
        if execute:
            if "df_session_state" not in st.session_state:
                st.session_state["df_session_state"] = df2.copy()
                st.session_state["df_session_state"][to_modify] = st.session_state["df_session_state"][to_modify].replace(modify_data, modify_word)
                df2[to_modify] = df2[to_modify].replace(modify_data, modify_word)
            else:
                st.session_state["df_session_state"][to_modify] = st.session_state["df_session_state"][to_modify].replace(modify_data, modify_word)
                df2[to_modify] = df2[to_modify].replace(modify_data, modify_word)
                

            st.write(st.session_state["df_session_state"])
            df2 = st.session_state["df_session_state"].copy()
    else:
        st.write("準識別子にカテゴリ変数がある場合にメニューを表示します")

    st.write("-----------------------------------")
    st.markdown("#### Step3 実行")
    next_step = st.button("匿名加工を実行する")
    
    if next_step :
        de_id = De_id(df2, unique_id, quasi_identifier, categorical, sensitive_attributes=None)
        anonymous = de_id.k_anonymize(k = k_value)
        df_anonymous = pd.DataFrame(anonymous)
        st.write(df_anonymous)
        
        #エントロピー
        st.write("情報量の変化")
        column_list = df_anonymous.columns.tolist()
        column_list.remove("count")
        
        entropy_dict = {}
        for col in column_list:
            freq = pd.Series.value_counts(df2[col])
            prob = freq / sum(freq)
            freq2 = pd.Series.value_counts(df_anonymous[col])
            prob2 = freq2 / sum(freq2)
            e = entropy(prob, base=2)
            e_2 = entropy(prob2, base=2)
            ent = [e, e_2]
            entropy_dict[col] = ent
        df_entropy = pd.DataFrame.from_dict(entropy_dict, orient="index", columns=["original","anonymous"])
        st.write(df_entropy)


        csv = df_anonymous.to_csv().encode('SHIFT-JIS')
        download = st.download_button(label='Data Download', data=csv, file_name='k_annonymous.csv',mime='text/csv')
    else :
        pass


