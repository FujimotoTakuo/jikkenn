# coding:utf-8

import MeCab
import sys

def mecab(text):
    m = MeCab.Tagger ("-Ochasen")

    # 頭にくっつけたり、後ろにくっつけたりする必要がある
    # そのまま出力はしない。くっつけの判定を終えて確定した前テキストから出力をする
    # まず、1行目は、前テキストに保持して終わり
    # 2行目以降は、
    #   1.前テキストが、頭にくっつける場合 : 前テキスト + 今テキスト　品詞は、今テキスト(今テキスト品詞を前テキスト品詞にする)
    #   2.今テキストが、後ろにくっつける場合 : 前テキスト + 今テキスト 品詞は、前テキスト
    #   ⇒  1 2は同時に出ないはず
    #   1, 2 にも該当しない場合 : 前テキストは、確定した。出力する。出力後、前テキスト情報に今テキスト情報を入れる
    # ループ終了 : 最終行が前テキストに入っているので、出力
    #

    kuttuke_list = ["名詞-and-接尾-and-一般-and-*",
                    "助詞-and-連体化-and-*-and-*",
                    "名詞-and-特殊-and-助動詞語幹-and-*",
                    "名詞-and-接尾-and-特殊-and-*",
                    "助詞-and-副助詞／並立助詞／終助詞-and-*-and-*",
                    "助詞-and-終助詞-and-*-and-*",
                    "名詞-and-接尾-and-助数詞-and-*",
                    "名詞-and-非自立-and-一般",
                    "動詞-and-接尾-and-*-and-*",
                    "名詞-and-接尾-and-助数詞-and-*",
                    "形容詞-and-非自立-and-*-and-*",
                    "助詞-and-格助詞-and-一般-and-*",
                    "助詞-and-係助詞-and-*-and-*",
                    "助詞-and-接続助詞-and-*-and-*",
                    "助詞-and-副助詞-and-*-and-*",
                    "助動詞-and-*-and-*-and-*",
                    "動詞-and-非自立-and-*-and-*",
                    "名詞-and-接尾-and-サ変接続-and-*",
                    "名詞-and-接尾-and-一般-and-*",
                    "名詞-and-接尾-and-形容動詞語幹-and-*",
                    "名詞-and-接尾-and-助数詞-and-*",
                    "名詞-and-接尾-and-助動詞語幹-and-*",
                    "名詞-and-接尾-and-副詞可能-and-*"]

    kuttuke_word_list = ["されている", "いただける", "しよう", "する", "させていただき", "しました",
                         "した", "します", "したい", "していくより", "していきたい", "して",
                         "いただけると", "し", "している", "ある", "して行くと", "してくれる",
                         "しょう", "したら", "しています", "してください", "しない", "いる",
                         "できないですし", "してしまいます", "したくても", "いるので", "つき", "せんでした",
                         "していて", "なり", "された", "される", "いません", "すればいい", "ません", "ませんでした",
                         "できました", "いました", "すると", "させる", "するし", "してます", "されました", "され", "されませんでした",
                         "しても", "していた", "すれば", "なってしまいそうで", "していなかっても", "していなくっても",
                         "させて頂いております", "させていただいております", "させていただきます", "しましますが", "しまいますが", "しそうに",
                         "していました", "しやすい", "してて", "しましたが", "しておくと", "するよ",
                         "してみた", "されなかった", "していますが", "しれませんが", "しており", "しております",
                         "するので", "されているので", "してる", "できるでしょう", "しておきます", "しちゃってる",
                         "するなら", "してたら", "し始め", "しろ", "してから", "されてき",
                         "されない", "しますが", "してくれました", "されます", "されてしまいました", "されましたが",
                         "してますが", "しておらず", "されるものの", "しないといけない", "してありました", "してない",
                         "したいです", "しても良い", "してもよい", "してもいい", "されず", "しているので",
                         "していないので", "しないので", "してくれたら", "しなくてはなりません", "しておりました", "しつつ",
                         "していたので", "しそうなので", "してしまい", "したので", "して下さい", "してください", "し",
                         "してきましたが", "したくないので", "してもらう", "さしたら", "させたら", "させて下さい", "では", "するもの", "するものです",
                         "させてください", "し終えた", "しおえた", "しておけば", "したが", "して欲しいです", "には", "で",
                         "してほしいです", "していく", "していると", "しなくて", "してしまう", "してい", "は", "が", "を", "すること",
                         "しながら", "しないのに", "してくる", "してあり", "しにくく", "しませんでした", "に", "の", "も", "しないこと",
                         "してた", "してある", "させていこう", "させていただこう", "してれば", "していれば", "だ", "へ",
                         "しておりません", "しなければならない", "しないで", "て", "って", "っ", "っていう", "た", "った", "だった"]

    atama_list = ["接頭詞-and-名詞接続-and-*-and-*"]

    hajikknai_kigou_list = ["記号-and-句点-and-*-and-*", "記号-and-読点-and-*-and-*"]

    result_node = m.parseToNode(text)
    mecabed_text = ""
    before_text = ""
    before_hinshi = ""
    while result_node:
        split_array = result_node.feature.split(",")
        if ('/' in split_array[0]) :
            result_node = result_node.next
            continue
        current_text = result_node.surface
        hinshi = split_array[0] + "-and-" + split_array[1] + "-and-" + split_array[2] + "-and-" + split_array[3]


        # 記号は基本はじくが、句読点ははじかない。
        # はじかないListに含まれない、記号はcontinue
        if '記号' in split_array[0] and not hinshi in hajikknai_kigou_list:
            result_node = result_node.next
            continue

        # 初回はブランク 保持して次へ
        if not before_text:
            before_text = current_text
            before_hinshi = hinshi
            result_node = result_node.next
            continue


        # 前のテキストが、頭に付けるかチェック
        if before_hinshi in atama_list and not hinshi in hajikknai_kigou_list:
            # 前のテキスト + 今のテキスト
            before_text = before_text + current_text
            # before_hinshiを変える
            before_hinshi = hinshi
            result_node = result_node.next
            continue

        # 今のテキストが、後ろにくっつける品詞かチェック
        if (hinshi in kuttuke_list or current_text in kuttuke_word_list) and not before_hinshi in hajikknai_kigou_list:
            # 前のテキスト + 今のテキスト　まだ単語の確定はしない before_hinshiは変えない。
            before_text = before_text + current_text
            result_node = result_node.next
            continue

        # before を書き込む
        mecabed_text = mecabed_text + ' ' + before_text

        # 今のテキストは保持して次へ
        before_text = current_text
        before_hinshi = hinshi

        result_node = result_node.next

    # before を書き込む
    mecabed_text = mecabed_text + ' ' + before_text
    # 最初のスペースを削除してreturn
    return mecabed_text[1:-1]

### Execute
def mecabFile(path):
    out_path = path + '_mecabed.txt'
    in_file = open(path, 'r')
    out_file = open(out_path, 'w')
    text_list = []

    counter = 0
    for line in in_file:
        counter += 1
        if counter % 100 == 0:
            print("count : " + str(counter))
        text_list.append(mecab(line))

    all_file = open(out_path, 'w')
    for  str_text in text_list:
        all_file.write(str_text + "\n")

    all_file.close()


if __name__ == "__main__":
    param = sys.argv
    path = param[1]
    mecabFile(path)
