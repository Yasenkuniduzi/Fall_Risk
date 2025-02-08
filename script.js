document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const toggleLanguageButton = document.getElementById('toggle-language');
    let isChinese = true;

    toggleLanguageButton.addEventListener('click', function() {
        isChinese = !isChinese;
        toggleLanguage(isChinese);
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = {};
        const formElements = form.elements;
        for (let element of formElements) {
            if (element.name && element.name !== '') {
                formData[element.name] = element.value;
            }
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            
            if (response.ok) {
                showResults(formData, result);
            } else {
                displayError(result.errors || result.error);
            }
        } catch (error) {
            displayError(['服务器连接错误，请稍后重试 / Server connection error, please try again later']);
        }
    });

    function toggleLanguage(isChinese) {
        const elements = document.querySelectorAll('[data-cn], [data-en]');
        elements.forEach(el => {
            el.textContent = isChinese ? el.getAttribute('data-cn') : el.getAttribute('data-en');
        });
    }

    function showResults(formData, result) {
        const modal = document.getElementById('result-modal');
        
        document.getElementById('result-name').textContent = formData.name;
        document.getElementById('result-age').textContent = formData.Age;
        document.getElementById('result-sex').textContent = formData.Sex === '1' ? '男 / Male' : '女 / Female';
        document.getElementById('result-marital').textContent = formData.Marital_status === '1' ? '已婚 / Married' : '其他 / Other';
        document.getElementById('result-residence').textContent = formData.Residence === '1' ? '城市 / Urban' : '农村 / Rural';
        
        document.getElementById('risk-2018').textContent = `${(result.risk_2018 * 100).toFixed(1)}%`;
        document.getElementById('risk-2020').textContent = `${(result.risk_2020 * 100).toFixed(1)}%`;
        
        const recommendationsList = document.getElementById('recommendations-list');
        recommendationsList.innerHTML = '';
        result.recommendations.forEach((recommendation, index) => {
            const div = document.createElement('div');
            div.className = 'recommendation-item';
            div.textContent = `${index + 1}. ${recommendation}`;
            recommendationsList.appendChild(div);
        });
        
        document.getElementById('report-date').textContent = 
            new Date().toLocaleDateString('zh-CN', { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        
        modal.style.display = 'block';
    }

    document.querySelector('.close').onclick = function() {
        document.getElementById('result-modal').style.display = 'none';
    }

    document.getElementById('close-modal').onclick = function() {
        document.getElementById('result-modal').style.display = 'none';
    }

    window.onclick = function(event) {
        const modal = document.getElementById('result-modal');
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }

    document.getElementById('download-pdf').onclick = async function() {
        const { jsPDF } = window.jspdf;
        const modalContent = document.querySelector('.modal-content');
        
        try {
            const style = document.createElement('style');
            style.textContent = `
                @media print {
                    .modal-actions { display: none !important; }
                    .close { display: none !important; }
                    .modal-content { box-shadow: none !important; }
                }
            `;
            document.head.appendChild(style);
            
            const modalActions = document.querySelector('.modal-actions');
            modalActions.style.display = 'none';
            
            const canvas = await html2canvas(modalContent, {
                scale: 2,
                useCORS: true,
                logging: false,
                removeContainer: true
            });
            
            const pdf = new jsPDF('p', 'mm', 'a4');
            const imgWidth = 210;
            const imgHeight = canvas.height * imgWidth / canvas.width;
            
            pdf.addImage(canvas.toDataURL('image/png'), 'PNG', 0, 0, imgWidth, imgHeight);
            
            const name = document.getElementById('result-name').textContent;
            const date = new Date().toISOString().slice(0, 10);
            pdf.save(`${name}_跌倒风险评估报告_${date}.pdf`);
            
            modalActions.style.display = 'flex';
            
            document.head.removeChild(style);
        } catch (error) {
            console.error('PDF生成失败:', error);
            alert('PDF生成失败，请稍后重试 / PDF generation failed, please try again later');
        }
    }

    document.getElementById('print-result').onclick = function() {
        window.print();
    }
});

document.getElementById('predictionForm').addEventListener('submit', function(event) {
    // 检查所有必填字段
    const requiredFields = [
        'name', 'Age', 'Sex', 'Marital_status', 'Residence', 'Smoking', 'Drinking',
        'Physical_disabilities', 'Mental_retardation', 'Vision_problem',
        'Hearing_problem', 'Speech_impediment', 'difficulty_with_getting_up',
        'difficulty_with_stooping'
    ];
    
    // 添加慢性病字段
    for (let i = 1; i <= 14; i++) {
        requiredFields.push(`new_da007_${i}_`);
    }
    
    let isValid = true;
    let missingFields = [];
    
    requiredFields.forEach(field => {
        const elements = document.getElementsByName(field);
        if (elements.length === 0) {
            isValid = false;
            missingFields.push(field);
        } else if (elements[0].type === 'radio') {
            // 对于单选按钮，检查是否有选中的选项
            const checked = Array.from(elements).some(el => el.checked);
            if (!checked) {
                isValid = false;
                missingFields.push(field);
            }
        } else if (!elements[0].value) {
            isValid = false;
            missingFields.push(field);
        }
    });
    
    if (!isValid) {
        event.preventDefault();
        alert('请填写所有必需的字段 / Please fill in all required fields\n' + 
              '缺少字段 / Missing fields: ' + missingFields.join(', '));
        return false;
    }
    
    // 验证年龄范围
    const age = parseInt(document.getElementsByName('Age')[0].value);
    if (isNaN(age) || age < 40 || age > 120) {
        event.preventDefault();
        alert('年龄必须在40-120岁之间 / Age must be between 40 and 120');
        return false;
    }
    
    // 如果所有验证都通过，允许表单提交
    return true;
});