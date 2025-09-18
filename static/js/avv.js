document.addEventListener('DOMContentLoaded', function() {
    const tabs = document.querySelectorAll('.tabs_nav_item');
    const panes = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs and hide all panes
            tabs.forEach(t => {
                t.classList.remove('is-active');
                t.setAttribute('aria-selected', 'false');
            });
            panes.forEach(p => {
                p.style.display = 'none';
            });

            // Add active class to the clicked tab and show corresponding pane
            this.classList.add('is-active');
            this.setAttribute('aria-selected', 'true');
            const paneId = this.getAttribute('aria-controls');
            document.getElementById(paneId).style.display = 'block';
        });
    });
});

// 提交页面

function handleFileUpload() {
    const fileInput = document.querySelector('input[type="file"]');
    const tabblistDiv = document.querySelector('.tabbable');
    const submitDiv = document.querySelector('.resubmit');
    const optionsDiv = document.querySelector('.options');
    const claimDiv = document.querySelector('.hawk-claim');
    const fileNameDiv = document.querySelector('.upload-filename');
    const filenameInput = document.querySelector('.file-names');

    if (fileInput.files.length > 0) {
        tabblistDiv.style.display = 'none'; 
        submitDiv.style.display = 'block';
        optionsDiv.style.display = 'block'; 
        fileNameDiv.style.display = 'flex';
        claimDiv.style.display = 'none';

        // 获取选定的文件名并显示
        const fileNames = Array.from(fileInput.files).map(file => file.name).join(', ');
        filenameInput.value = fileNames; // 显示文件名
    } else {
        filenameInput.value = '未选定任何文件'; // 如果没有文件，显示提示
    }
}


document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('toggle-menu').addEventListener('click', function() {
        const downIcon = this.querySelector('.bi-chevron-down');
        const upIcon = this.querySelector('.bi-chevron-up');
        const menuMore = document.querySelector('.menu_more');

        // 切换图标和菜单显示状态
        if (menuMore.style.display === 'none') {
            menuMore.style.display = 'block';
            downIcon.style.display = 'none';
            upIcon.style.display = 'inline';
        } else {
            menuMore.style.display = 'none';
            downIcon.style.display = 'inline';
            upIcon.style.display = 'none';
        }
    });
});

// 切换显示hawk引擎

// 定义切换函数
function toggleDivs(show) {
    const switchIcons = document.querySelectorAll('.toggle-menu i');
    const hawkItems = document.querySelector('.deeplearning-items');
    const hawkDetails = document.querySelector('.hawk-details');

    if (show) {
        switchIcons[0].style.display = 'none';
        switchIcons[1].style.display = 'inline-block';
        hawkItems.style.display = 'none';
        hawkDetails.style.display = 'block';
    } else {
        switchIcons[0].style.display = 'inline-block';
        switchIcons[1].style.display = 'none';
        hawkItems.style.display = 'flex';
        hawkDetails.style.display = 'none';
    }
}

// 为链接添加鼠标移入和移出事件
const toggleMenu = document.querySelector('.toggle-menu');
toggleMenu.addEventListener('mouseenter', () => toggleDivs(true));
toggleMenu.addEventListener('mouseleave', () => toggleDivs(false));


