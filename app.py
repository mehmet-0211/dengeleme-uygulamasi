import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Dengeleme Uygulaması", layout="wide")

st.title("Dengeleme Uygulaması")
st.write("Nivelman, koşullu ve dolaylı dengeleme hesapları için açıklamalı arayüz.")

def hata_analizi(v, P, serbestlik):
    vtpv = float((v.T @ P @ v)[0, 0])

    if serbestlik > 0:
        sigma0 = float(np.sqrt(vtpv / serbestlik))
    else:
        sigma0 = None

    mutlak = np.abs(v).flatten()
    max_index = int(np.argmax(mutlak))
    max_deger = float(mutlak[max_index])

    return vtpv, sigma0, max_index, max_deger

def rapor_metni(baslik, icerik):
    tarih = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    return f"{baslik}\n{'=' * len(baslik)}\nTarih: {tarih}\n\n{icerik}"

secim = st.sidebar.selectbox(
    "Dengeleme türünü seç",
    ["Nivelman dengelemesi", "Koşullu dengeleme", "Dolaylı dengeleme"]
)

# ======================================================
# 1 - NİVELMAN
# ======================================================

if secim == "Nivelman dengelemesi":
    st.header("Nivelman Dengelemesi")
    st.info("Bu modda B matrisi otomatik kurulur. Kullanıcı sadece kotları, yükseklik farklarını ve ağırlıkları girer.")

    col1, col2, col3 = st.columns(3)

    with col1:
        baslangic = st.number_input("Başlangıç kotu", value=100.0, format="%.4f")

    with col2:
        bitis = st.number_input("Bitiş kotu", value=101.0, format="%.4f")

    with col3:
        n = st.number_input("Yükseklik farkı sayısı", min_value=1, value=4, step=1)

    st.subheader("Ölçüler ve Ağırlıklar")

    olcumler = []
    agirliklar = []

    for i in range(int(n)):
        c1, c2 = st.columns(2)

        with c1:
            olcum = st.number_input(
                f"{i+1}. yükseklik farkı",
                value=0.0,
                format="%.6f",
                key=f"nivelman_olcum_{i}"
            )

        with c2:
            agirlik = st.number_input(
                f"{i+1}. ağırlık",
                min_value=0.0001,
                value=1.0,
                format="%.4f",
                key=f"nivelman_agirlik_{i}"
            )

        olcumler.append(olcum)
        agirliklar.append(agirlik)

    if st.button("Nivelman Hesapla"):
        l = np.array(olcumler, dtype=float).reshape(int(n), 1)
        P = np.diag(agirliklar)

        teorik_fark = bitis - baslangic
        toplam_olcum = float(np.sum(l))
        kapanma_hatasi = toplam_olcum - teorik_fark
        f = np.array([[kapanma_hatasi]], dtype=float)

        B = np.ones((1, int(n)), dtype=float)

        P_inv = np.linalg.inv(P)
        N = B @ P_inv @ B.T
        k = np.linalg.solve(N, f)
        v = -P_inv @ B.T @ k
        l_duz = l + v

        kotlar = [baslangic]
        for i in range(int(n)):
            kotlar.append(kotlar[-1] + l_duz[i, 0])

        serbestlik = int(n) - 1
        vtpv, sigma0, max_index, max_deger = hata_analizi(v, P, serbestlik)

        st.subheader("Sonuçlar")

        c1, c2, c3 = st.columns(3)
        c1.metric("Toplam ölçülen fark", f"{toplam_olcum:.6f}")
        c2.metric("Teorik fark", f"{teorik_fark:.6f}")
        c3.metric("Kapanma hatası", f"{kapanma_hatasi:.6f}")

        tablo = pd.DataFrame({
            "No": list(range(1, int(n) + 1)),
            "Ham Fark": l.flatten(),
            "Ağırlık": agirliklar,
            "Düzeltme": v.flatten(),
            "Düzeltilmiş Fark": l_duz.flatten(),
            "Kot": kotlar[1:]
        })

        st.subheader("Nivelman Tablosu")
        st.dataframe(tablo, use_container_width=True)

        st.subheader("Matrisler")
        with st.expander("B matrisi"):
            st.write(B)

        with st.expander("P matrisi"):
            st.write(P)

        with st.expander("N = B P⁻¹ Bᵀ"):
            st.write(N)

        st.subheader("Kontrol")
        st.write("Bv + f:")
        st.write(B @ v + f)

        st.subheader("Hata Analizi")
        st.write(f"Serbestlik derecesi: {serbestlik}")
        st.write(f"vᵀPv: {vtpv:.6f}")

        if sigma0 is not None:
            st.write(f"sigma0: {sigma0:.6f}")
        else:
            st.warning("sigma0 hesaplanamadı.")

        st.write(f"En büyük düzeltme: {max_index + 1}. ölçü → {max_deger:.6f}")

        st.subheader("Açıklama")
        if kapanma_hatasi > 0:
            st.write("Kapanma hatası pozitif olduğu için ölçülere genel olarak negatif düzeltme verilmiştir.")
        elif kapanma_hatasi < 0:
            st.write("Kapanma hatası negatif olduğu için ölçülere genel olarak pozitif düzeltme verilmiştir.")
        else:
            st.write("Kapanma hatası sıfırdır.")

        st.write("Ağırlığı büyük olan ölçüler daha güvenilir kabul edilir ve genelde daha az düzeltme alır.")

        rapor = ""
        rapor += f"Başlangıç kotu: {baslangic}\n"
        rapor += f"Bitiş kotu: {bitis}\n"
        rapor += f"Toplam ölçülen fark: {toplam_olcum:.6f}\n"
        rapor += f"Teorik fark: {teorik_fark:.6f}\n"
        rapor += f"Kapanma hatası: {kapanma_hatasi:.6f}\n\n"
        rapor += tablo.to_string(index=False)
        rapor += f"\n\nvTPv: {vtpv:.6f}\n"
        rapor += f"sigma0: {sigma0}\n"
        rapor += f"En büyük düzeltme: {max_index + 1}. ölçü -> {max_deger:.6f}\n"

        st.download_button(
            "Raporu indir",
            data=rapor_metni("NİVELMAN DENGELEME RAPORU", rapor),
            file_name="nivelman_rapor.txt",
            mime="text/plain"
        )

# ======================================================
# 2 - KOŞULLU
# ======================================================

elif secim == "Koşullu dengeleme":
    st.header("Koşullu Dengeleme")
    st.info("Model: Bv + f = 0. Bu modda B matrisi ve f vektörü kullanıcı tarafından girilir.")

    c1, c2 = st.columns(2)

    with c1:
        n = st.number_input("Ölçü sayısı (n)", min_value=1, value=4, step=1)

    with c2:
        m = st.number_input("Koşul sayısı (m)", min_value=1, value=2, step=1)

    st.subheader("Ölçüler ve Ağırlıklar")

    olcumler = []
    agirliklar = []

    for i in range(int(n)):
        c1, c2 = st.columns(2)

        with c1:
            olcumler.append(
                st.number_input(f"{i+1}. ölçü", value=0.0, format="%.6f", key=f"kosullu_olcum_{i}")
            )

        with c2:
            agirliklar.append(
                st.number_input(f"{i+1}. ağırlık", min_value=0.0001, value=1.0, format="%.4f", key=f"kosullu_agirlik_{i}")
            )

    st.subheader("B Matrisi")
    B_rows = []

    for i in range(int(m)):
        satir = st.text_input(
            f"{i+1}. koşul katsayıları ({int(n)} sayı, boşlukla):",
            value=" ".join(["0"] * int(n)),
            key=f"B_satir_{i}"
        )

        try:
            degerler = [float(x) for x in satir.split()]
        except ValueError:
            degerler = []

        B_rows.append(degerler)

    st.subheader("f Vektörü")
    f_list = []

    for i in range(int(m)):
        f_list.append(
            st.number_input(f"{i+1}. f değeri", value=0.0, format="%.6f", key=f"f_{i}")
        )

    if st.button("Koşullu Hesapla"):
        try:
            if any(len(row) != int(n) for row in B_rows):
                st.error("B matrisindeki her satır ölçü sayısı kadar katsayı içermelidir.")
            else:
                l = np.array(olcumler, dtype=float).reshape(int(n), 1)
                P = np.diag(agirliklar)
                B = np.array(B_rows, dtype=float)
                f = np.array(f_list, dtype=float).reshape(int(m), 1)

                if np.linalg.matrix_rank(B) < int(m):
                    st.error("B matrisi tam satır rankına sahip değil.")
                else:
                    P_inv = np.linalg.inv(P)
                    N = B @ P_inv @ B.T
                    k = np.linalg.solve(N, f)
                    v = -P_inv @ B.T @ k
                    l_duz = l + v

                    serbestlik = int(n) - int(m)
                    vtpv, sigma0, max_index, max_deger = hata_analizi(v, P, serbestlik)

                    st.subheader("Sonuçlar")

                    tablo = pd.DataFrame({
                        "No": list(range(1, int(n) + 1)),
                        "Ham Ölçü": l.flatten(),
                        "Ağırlık": agirliklar,
                        "Düzeltme": v.flatten(),
                        "Düzeltilmiş Ölçü": l_duz.flatten()
                    })

                    st.dataframe(tablo, use_container_width=True)

                    st.subheader("Matrisler")
                    with st.expander("B matrisi"):
                        st.write(B)
                    with st.expander("N = B P⁻¹ Bᵀ"):
                        st.write(N)
                    with st.expander("k = N⁻¹ f"):
                        st.write(k)

                    st.subheader("Kontrol")
                    st.write("Bv + f:")
                    st.write(B @ v + f)

                    st.subheader("Hata Analizi")
                    st.write(f"Serbestlik derecesi: {serbestlik}")
                    st.write(f"vᵀPv: {vtpv:.6f}")
                    st.write(f"sigma0: {sigma0}")
                    st.write(f"En büyük düzeltme: {max_index + 1}. ölçü → {max_deger:.6f}")

                    st.subheader("Açıklama")
                    st.write("Koşullu dengelemede bilinmeyenler doğrudan ölçü düzeltmeleridir.")
                    st.write("Bv + f sonucu sıfıra yakınsa koşullar sağlanmıştır.")

                    rapor = ""
                    rapor += "B matrisi:\n" + str(B) + "\n\n"
                    rapor += "Düzeltmeler:\n" + str(v) + "\n\n"
                    rapor += "Kontrol Bv + f:\n" + str(B @ v + f) + "\n\n"
                    rapor += tablo.to_string(index=False)
                    rapor += f"\n\nvTPv: {vtpv:.6f}\nsigma0: {sigma0}\n"

                    st.download_button(
                        "Raporu indir",
                        data=rapor_metni("KOŞULLU DENGELEME RAPORU", rapor),
                        file_name="kosullu_rapor.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# ======================================================
# 3 - DOLAYLI
# ======================================================

elif secim == "Dolaylı dengeleme":
    st.header("Dolaylı Dengeleme")
    st.info("Model: Ax = l. Bu modda A matrisi, l vektörü ve ağırlıklar girilir.")

    c1, c2 = st.columns(2)

    with c1:
        m = st.number_input("Denklem sayısı (m)", min_value=1, value=3, step=1)

    with c2:
        n = st.number_input("Bilinmeyen sayısı (n)", min_value=1, value=2, step=1)

    st.subheader("A Matrisi")

    A_rows = []

    for i in range(int(m)):
        satir = st.text_input(
            f"{i+1}. A satırı ({int(n)} sayı, boşlukla):",
            value=" ".join(["0"] * int(n)),
            key=f"A_satir_{i}"
        )

        try:
            degerler = [float(x) for x in satir.split()]
        except ValueError:
            degerler = []

        A_rows.append(degerler)

    st.subheader("l Vektörü ve Ağırlıklar")

    l_list = []
    agirliklar = []

    for i in range(int(m)):
        c1, c2 = st.columns(2)

        with c1:
            l_list.append(
                st.number_input(f"{i+1}. l değeri", value=0.0, format="%.6f", key=f"l_{i}")
            )

        with c2:
            agirliklar.append(
                st.number_input(f"{i+1}. ağırlık", min_value=0.0001, value=1.0, format="%.4f", key=f"dolayli_agirlik_{i}")
            )

    if st.button("Dolaylı Hesapla"):
        try:
            if any(len(row) != int(n) for row in A_rows):
                st.error("A matrisindeki her satır bilinmeyen sayısı kadar katsayı içermelidir.")
            elif int(m) < int(n):
                st.error("Denklem sayısı bilinmeyen sayısından küçük olamaz.")
            else:
                A = np.array(A_rows, dtype=float)
                l = np.array(l_list, dtype=float).reshape(int(m), 1)
                P = np.diag(agirliklar)

                N = A.T @ P @ A
                n_vec = A.T @ P @ l

                if np.linalg.matrix_rank(N) < int(n):
                    st.error("N = AᵀPA matrisi terslenebilir değil. Rank problemi var.")
                else:
                    x = np.linalg.solve(N, n_vec)
                    l_hat = A @ x
                    v = l_hat - l

                    serbestlik = int(m) - int(n)
                    vtpv, sigma0, max_index, max_deger = hata_analizi(v, P, serbestlik)

                    st.subheader("Sonuçlar")

                    x_tablo = pd.DataFrame({
                        "Bilinmeyen": [f"x{i+1}" for i in range(int(n))],
                        "Değer": x.flatten()
                    })

                    st.dataframe(x_tablo, use_container_width=True)

                    tablo = pd.DataFrame({
                        "No": list(range(1, int(m) + 1)),
                        "l": l.flatten(),
                        "l şapka": l_hat.flatten(),
                        "Artık v": v.flatten(),
                        "Ağırlık": agirliklar
                    })

                    st.subheader("Gözlem Tablosu")
                    st.dataframe(tablo, use_container_width=True)

                    st.subheader("Matrisler")
                    with st.expander("A matrisi"):
                        st.write(A)
                    with st.expander("N = Aᵀ P A"):
                        st.write(N)
                    with st.expander("n = Aᵀ P l"):
                        st.write(n_vec)

                    st.subheader("Kontrol")
                    st.write("Ax - l:")
                    st.write(A @ x - l)

                    st.subheader("Hata Analizi")
                    st.write(f"Serbestlik derecesi: {serbestlik}")
                    st.write(f"vᵀPv: {vtpv:.6f}")
                    st.write(f"sigma0: {sigma0}")
                    st.write(f"En büyük artık: {max_index + 1}. gözlem → {max_deger:.6f}")

                    st.subheader("Açıklama")
                    st.write("Dolaylı dengelemede ana hedef bilinmeyen x değerlerini bulmaktır.")
                    st.write("Artıklar, bulunan x değerlerinin gözlemlerle ne kadar uyumlu olduğunu gösterir.")

                    rapor = ""
                    rapor += "A matrisi:\n" + str(A) + "\n\n"
                    rapor += "x bilinmeyenleri:\n" + str(x) + "\n\n"
                    rapor += "Artıklar:\n" + str(v) + "\n\n"
                    rapor += "Kontrol Ax - l:\n" + str(A @ x - l) + "\n\n"
                    rapor += tablo.to_string(index=False)
                    rapor += f"\n\nvTPv: {vtpv:.6f}\nsigma0: {sigma0}\n"

                    st.download_button(
                        "Raporu indir",
                        data=rapor_metni("DOLAYLI DENGELEME RAPORU", rapor),
                        file_name="dolayli_rapor.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
